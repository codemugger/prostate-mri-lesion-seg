"""CARPL ai-worker: preflight, selection record, send gates and job lifecycle.

The real inference app is replaced by a tiny fake ``__main__.py`` so these
tests run anywhere.  The worker invokes it exactly as it would the real app
(``python <app_dir> -i <input> -o <output> -m <models>``).
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from pydicom.uid import generate_uid

import carpl_test_utils as utils
import carpl_common as cc
import case_status

worker_mod = utils.load_module("carpl_worker", "carpl/ai-worker/worker.py")

REAL_APP_PY = utils.ROOT / "prostate_mri_lesion_seg_app" / "app.py"

FAKE_MAIN = r'''
import os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["FAKE_TESTS_DIR"])
args = sys.argv[1:]
inp, out = Path(args[args.index("-i") + 1]), Path(args[args.index("-o") + 1])
mode = os.environ.get("FAKE_APP_MODE", "complete")
if mode == "sleep":
    time.sleep(60)
elif mode == "fail":
    sys.exit(3)
elif mode == "complete":
    import pydicom
    import carpl_test_utils as utils
    first = sorted(inp.rglob("*.dcm"))[0]
    ds = pydicom.dcmread(first, stop_before_pixels=True)
    utils.create_complete_output(out, str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), str(ds.PatientID))
'''


def make_models(root: Path) -> Path:
    for rel in worker_mod.EXPECTED_MODELS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
    return root


class PreflightTests(unittest.TestCase):

    def test_exact_model_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = make_models(Path(tmp) / "models")
            self.assertEqual(worker_mod.check_models(models), [])

            (models / "deployment_manifest.json").write_text("{}")
            (models / "organ" / ".DS_Store").write_text("")
            (models / "fold3" / "model_best_fold3.pth.tar").unlink()
            problems = worker_mod.check_models(models)
            self.assertTrue(any("deployment_manifest.json" in p for p in problems), problems)
            self.assertTrue(any(".DS_Store" in p for p in problems), problems)
            self.assertTrue(any("fold3" in p for p in problems), problems)

    def test_manifest_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = make_models(Path(tmp) / "models")
            manifest = Path(tmp) / "config" / "deployment_manifest.json"
            self.assertEqual(worker_mod.verify_manifest(models, manifest, required=False), [])
            self.assertTrue(worker_mod.verify_manifest(models, manifest, required=True))
            entries = {
                rel: {"sha256": hashlib.sha256(rel.encode()).hexdigest(), "bytes": len(rel)}
                for rel in worker_mod.EXPECTED_MODELS
            }
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(entries))
            self.assertEqual(worker_mod.verify_manifest(models, manifest, required=True), [])
            (models / "organ" / "model.ts").write_bytes(b"tampered")
            problems = worker_mod.verify_manifest(models, manifest, required=True)
            self.assertEqual(len(problems), 1)
            self.assertIn("organ/model.ts", problems[0])


class SelectionRuleTests(unittest.TestCase):
    """Uses the app's own rules (read, never imported) with SDK 2.0 matching."""

    def setUp(self):
        self.rules = worker_mod.load_selection_rules(REAL_APP_PY)

    def roles(self, description):
        attrs = {"Modality": "MR", "SeriesDescription": description}
        return sorted(role for role, cond in self.rules.items() if worker_mod.sdk_matches(cond, attrs))

    def test_rules_are_read_from_app_py(self):
        self.assertEqual(sorted(self.rules), ["adc", "highb", "t2"])
        self.assertEqual(self.rules["t2"]["Modality"], "MR")

    def test_prostatex_0004_series(self):
        self.assertEqual(self.roles("t2_tse_tra"), ["t2"])
        self.assertEqual(self.roles("t2_tse_sag"), [])
        self.assertEqual(self.roles("t2_tse_cor"), [])
        self.assertEqual(self.roles("ep2d_diff_tra_DYNDIST_MIX_ADC"), ["adc"])
        # Two HIGHB candidates: the pipeline keeps its first-match behaviour
        # and the worker records the ambiguity.
        self.assertEqual(self.roles("ep2d_diff_tra_DYNDIST_MIX"), ["highb"])
        self.assertEqual(self.roles("ep2d_diff_tra_DYNDIST_MIXCALC_BVAL"), ["highb"])
        self.assertEqual(self.roles("tfl_3d PD ref_tra_1.5x1.5_t3"), [])
        self.assertEqual(self.roles(""), [])

    def test_non_mr_never_matches(self):
        attrs = {"Modality": "CT", "SeriesDescription": "t2_tse_tra"}
        self.assertFalse(worker_mod.sdk_matches(self.rules["t2"], attrs))


class WorkerLifecycleTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.paths = cc.DataPaths(tmp / "data")
        self.paths.ensure()
        self.app_dir = tmp / "fake_app"
        self.app_dir.mkdir()
        (self.app_dir / "__main__.py").write_text(FAKE_MAIN)
        shutil.copy2(REAL_APP_PY, self.app_dir / "app.py")
        self.models = make_models(tmp / "models")
        self.broker = cc.InMemoryBroker()
        os.environ["FAKE_TESTS_DIR"] = str(Path(__file__).resolve().parent)
        os.environ["FAKE_APP_MODE"] = "complete"
        self.worker = self._worker()

    def tearDown(self):
        os.environ.pop("FAKE_APP_MODE", None)
        self._tmp.cleanup()

    def _worker(self, **overrides):
        config = worker_mod.WorkerConfig(
            app_dir=self.app_dir, app_workdir=self.app_dir, models_dir=self.models,
            require_gpu=False, job_timeout_seconds=overrides.pop("timeout", 60), **overrides,
        )
        return worker_mod.Worker(config, self.paths, self.broker, logging.getLogger("test"))

    def _queue_job(self, descriptions=("t2_tse_tra", "ep2d_diff_tra_ADC", "ep2d_diff_tra")):
        study = generate_uid()
        job_id = cc.new_job_id()
        job = self.paths.job(job_id)
        series = []
        for description in descriptions:
            series_uid = generate_uid()
            utils.write(utils.make_mr(study, series_uid, description=description), job.input / series_uid / "1.dcm")
            series.append({"series_instance_uid": series_uid, "modality": "MR", "series_description": description})
        fingerprint = cc.study_fingerprint([job_id])
        cc.atomic_write_json(job.job_json, {
            "job_id": job_id, "study_instance_uid": study, "fingerprint": fingerprint, "series": series,
        })
        cc.update_job_state(job, "queued")
        self.broker.set(cc.study_latest_key(study), job_id)
        return {"job_id": job_id, "study_uid": study, "fingerprint": fingerprint}, job

    def _fp_state(self, payload):
        return json.loads(self.broker.get(cc.fingerprint_key(payload["fingerprint"])))["state"]

    def test_successful_job_queues_both_results(self):
        payload, job = self._queue_job()
        self.worker.process(payload)
        self.assertEqual(cc.job_state(job), "send_queued")
        send = self.broker.items(cc.SEND)
        self.assertEqual(len(send), 1)
        self.assertEqual([Path(f).name for f in send[0]["files"]], list(cc.RESULT_FILES))
        self.assertEqual(self._fp_state(payload), "send_queued")
        rows = self.paths.audit_csv.read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn(case_status.Status.SUCCESS_COMPLETE, rows[1])
        self.assertIn(payload["job_id"], rows[1])
        selection = cc.read_json(job.root / "selection.json")
        self.assertEqual(selection["ambiguous_roles"], [])
        self.assertIn("-i", job.pipeline_log.read_text())

    def test_empty_output_is_not_sent(self):
        os.environ["FAKE_APP_MODE"] = "empty"
        payload, job = self._queue_job()
        self.worker.process(payload)
        self.assertEqual(cc.job_state(job), "failed")
        self.assertEqual(self.broker.length(cc.SEND), 0)
        self.assertEqual(self._fp_state(payload), "failed")
        self.assertIn(case_status.Status.FAIL_EMPTY_OUTPUT, self.paths.audit_csv.read_text())

    def test_non_zero_exit_is_runtime_failure(self):
        os.environ["FAKE_APP_MODE"] = "fail"
        payload, job = self._queue_job()
        self.worker.process(payload)
        self.assertEqual(cc.job_state(job), "failed")
        self.assertIn(case_status.Status.FAIL_RUNTIME, self.paths.audit_csv.read_text())

    def test_timeout_kills_the_pipeline(self):
        os.environ["FAKE_APP_MODE"] = "sleep"
        worker = self._worker(timeout=1)
        payload, job = self._queue_job()
        worker.process(payload)
        self.assertEqual(cc.job_state(job), "failed")
        self.assertIn("JOB_TIMEOUT_SECONDS", cc.read_json(job.state_json)["reason"])
        self.assertEqual(self.broker.length(cc.SEND), 0)

    def test_superseded_job_is_skipped_before_inference(self):
        payload, job = self._queue_job()
        self.broker.set(cc.study_latest_key(payload["study_uid"]), "newer-job")
        self.worker.process(payload)
        self.assertEqual(cc.job_state(job), "superseded")
        self.assertFalse(job.pipeline_log.exists())

    def test_recovered_after_inference_is_not_rerun(self):
        payload, job = self._queue_job()
        self.worker.process(payload)
        self.broker = cc.InMemoryBroker()  # e.g. worker crashed before the send was queued
        worker = self._worker()
        cc.update_job_state(job, "inferred")
        log_size = job.pipeline_log.stat().st_size
        os.environ["FAKE_APP_MODE"] = "fail"  # would break the job if it were re-run
        worker.process(payload)
        self.assertEqual(cc.job_state(job), "send_queued")
        self.assertEqual(self.broker.length(cc.SEND), 1)
        self.assertEqual(job.pipeline_log.stat().st_size, log_size, "pipeline must not run again")

    def test_attempt_limit(self):
        payload, job = self._queue_job()
        cc.update_job_state(job, "running", attempts=3)
        self.worker.process(payload)
        self.assertEqual(cc.job_state(job), "failed")
        self.assertEqual(self.broker.length(cc.JOBS_FAILED), 1)

    def test_ambiguous_selection_policy(self):
        descriptions = ("t2_tse_tra", "ep2d_diff_tra_ADC", "ep2d_diff_tra", "calculated b1400")
        payload, job = self._queue_job(descriptions)
        self.worker.process(payload)
        self.assertEqual(cc.job_state(job), "send_queued", "default policy keeps existing behaviour")
        self.assertEqual(cc.read_json(job.root / "selection.json")["ambiguous_roles"], ["highb"])

        holding = self._worker(ambiguous_selection_policy="hold")
        payload, job = self._queue_job(descriptions)
        holding.process(payload)
        self.assertEqual(cc.job_state(job), "held")

    def test_janitor_purges_by_state_and_age(self):
        delivered, delivered_job = self._queue_job()
        cc.update_job_state(delivered_job, "delivered")
        running, running_job = self._queue_job()
        cc.update_job_state(running_job, "running")
        worker = self._worker(delivered_retention_hours=0)
        self.assertEqual(worker.janitor(now=10 ** 10), 1)
        self.assertFalse(delivered_job.root.exists())
        self.assertTrue(running_job.root.exists(), "active jobs are never purged")


class DecideSendTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.job = cc.JobPaths(Path(self._tmp.name) / "job")
        self.study, self.t2 = generate_uid(), generate_uid()
        utils.create_complete_output(self.job.output, self.study, self.t2)
        self.manifest = {"study_instance_uid": self.study, "series": [{"series_instance_uid": self.t2}]}
        self.config = worker_mod.WorkerConfig(require_gpu=False)
        self.selection = {"ambiguous_roles": []}

    def tearDown(self):
        self._tmp.cleanup()

    def decide(self):
        case = case_status.inspect_case(self.job.output)
        return worker_mod.decide_send(self.job, self.manifest, case, self.selection, self.config)

    def test_complete_output_is_sent(self):
        self.assertEqual(self.decide(), ("send", []))

    def test_wrong_study_is_never_sent(self):
        utils.write_sr(self.job.output / "prostate_measurements_SR.dcm", generate_uid(), "SYNTHETIC-001")
        decision, reasons = self.decide()
        self.assertEqual(decision, "fail")
        self.assertTrue(any("StudyInstanceUID" in r for r in reasons), reasons)

    def test_wrong_patient_is_never_sent(self):
        utils.write_sr(self.job.output / "prostate_measurements_SR.dcm", self.study, "SOMEONE-ELSE")
        decision, reasons = self.decide()
        self.assertEqual(decision, "fail")
        self.assertTrue(any("PatientID" in r for r in reasons), reasons)

    def test_legacy_rtstruct_is_never_sent(self):
        rtstruct = self.job.output / "combined_organ_lesion_RTSTRUCT.dcm"
        import pydicom
        ds = pydicom.dcmread(rtstruct)
        ds.StructureSetLabel = "RTstruct"
        ds.save_as(rtstruct)
        self.assertEqual(self.decide()[0], "fail")

    def test_empty_result_policy(self):
        (self.job.output / "lesion" / "merged_lesion_prob.nii.gz").unlink()
        self.assertEqual(self.decide()[0], "send")
        self.config = worker_mod.WorkerConfig(require_gpu=False, send_empty_results=False)
        self.assertEqual(self.decide()[0], "hold")


if __name__ == "__main__":
    unittest.main()
