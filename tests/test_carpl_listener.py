"""CARPL dicom-listener: study-level triggering, layout, dedup and recovery."""

import json
import logging
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pydicom
from pydicom.uid import generate_uid

import carpl_test_utils as utils
import carpl_common as cc

listener_mod = utils.load_module("carpl_listener", "carpl/dicom-listener/listener.py")

QUIET = 60.0


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def fake_assoc(ae_title="CARPL_TEST"):
    return SimpleNamespace(requestor=SimpleNamespace(ae_title=ae_title))


def store_event(ds, assoc):
    encoded = utils.encode(ds)
    return SimpleNamespace(assoc=assoc, dataset=ds, encoded_dataset=lambda: encoded)


class FlakyBroker(cc.InMemoryBroker):
    """Fails the first push, like a transient Redis outage."""

    def __init__(self):
        super().__init__()
        self.failures = 1

    def push(self, queue, payload):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("redis unavailable")
        super().push(queue, payload)


class ListenerLogicTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = cc.DataPaths(Path(self._tmp.name))
        self.paths.ensure()
        self.clock = FakeClock()
        self.broker = cc.InMemoryBroker()
        self.config = listener_mod.ListenerConfig(quiet_seconds=QUIET, min_free_gb=0)
        self.listener = self._make_listener(self.broker)
        self.study = generate_uid()
        self.series = [generate_uid(), generate_uid()]

    def tearDown(self):
        self._tmp.cleanup()

    def _make_listener(self, broker):
        return listener_mod.StudyListener(
            self.config, self.paths, broker, logging.getLogger("test"), clock=self.clock
        )

    def _send(self, listener, count_per_series=2, assoc=None, close=True, series=None, modality="MR"):
        assoc = assoc or fake_assoc()
        datasets = []
        for series_uid in series or self.series:
            for number in range(1, count_per_series + 1):
                ds = utils.make_mr(self.study, series_uid, instance_number=number, modality=modality)
                self.assertEqual(listener.handle_store(store_event(ds, assoc)), 0x0000)
                datasets.append(ds)
        if close:
            listener.handle_conn_close(SimpleNamespace(assoc=assoc))
        return datasets

    def _dispatch_after_quiet(self, listener):
        self.clock.now += QUIET + 1
        return listener.dispatch_once()

    def test_series_per_folder_layout_and_study_level_dispatch(self):
        datasets = self._send(self.listener)
        for ds in datasets:
            stored = self.paths.studies / self.study / ds.SeriesInstanceUID / f"{ds.SOPInstanceUID}.dcm"
            self.assertEqual(stored.read_bytes(), utils.encode(ds), "bytes must be stored as received")

        self.clock.now += QUIET - 5
        self.assertEqual(self.listener.dispatch_once(), [], "must not dispatch before the quiet period")

        jobs = self._dispatch_after_quiet(self.listener)
        self.assertEqual(len(jobs), 1)
        job = self.paths.job(jobs[0])
        manifest = cc.read_json(job.job_json)
        self.assertEqual(manifest["study_instance_uid"], self.study)
        self.assertEqual(manifest["instance_count"], 4)
        self.assertEqual(sorted(s["series_instance_uid"] for s in manifest["series"]), sorted(self.series))
        for series_uid in self.series:
            self.assertEqual(len(list((job.input / series_uid).glob("*.dcm"))), 2)
        self.assertEqual(cc.job_state(job), "queued")
        self.assertEqual(self.broker.items(cc.JOBS)[0]["job_id"], jobs[0])
        self.assertEqual(self.broker.get(cc.study_latest_key(self.study)), jobs[0])
        self.assertFalse((self.paths.studies / self.study / listener_mod.PENDING_MARKER).exists())

    def test_open_association_blocks_dispatch(self):
        assoc = fake_assoc()
        self._send(self.listener, assoc=assoc, close=False)
        self.clock.now += QUIET * 10
        self.assertEqual(self.listener.dispatch_once(), [], "an open association means the study may be incomplete")
        self.listener.handle_conn_close(SimpleNamespace(assoc=assoc))
        self.assertEqual(self.listener.dispatch_once(), [], "quiet period restarts when the association closes")
        self.assertEqual(len(self._dispatch_after_quiet(self.listener)), 1)

    def test_identical_resend_is_skipped(self):
        datasets = self._send(self.listener)
        self.assertEqual(len(self._dispatch_after_quiet(self.listener)), 1)
        assoc = fake_assoc()
        for ds in datasets:  # CARPL retransmits the same instances
            self.listener.handle_store(store_event(ds, assoc))
        self.listener.handle_conn_close(SimpleNamespace(assoc=assoc))
        self.assertEqual(self._dispatch_after_quiet(self.listener), [])
        self.assertEqual(self.broker.length(cc.JOBS), 1)

    def test_late_instances_create_a_superseding_job_over_the_full_set(self):
        self._send(self.listener)
        first = self._dispatch_after_quiet(self.listener)[0]
        self._send(self.listener, count_per_series=1, series=[generate_uid()])
        second = self._dispatch_after_quiet(self.listener)[0]
        self.assertNotEqual(first, second)
        self.assertEqual(cc.read_json(self.paths.job(second).job_json)["instance_count"], 5)
        self.assertEqual(self.broker.get(cc.study_latest_key(self.study)), second)

    def test_failed_fingerprint_allows_retry(self):
        datasets = self._send(self.listener)
        first = self._dispatch_after_quiet(self.listener)[0]
        fingerprint = cc.read_json(self.paths.job(first).job_json)["fingerprint"]
        self.broker.set(cc.fingerprint_key(fingerprint), json.dumps({"state": "failed", "job_id": first}))
        assoc = fake_assoc()
        for ds in datasets:
            self.listener.handle_store(store_event(ds, assoc))
        self.listener.handle_conn_close(SimpleNamespace(assoc=assoc))
        retry = self._dispatch_after_quiet(self.listener)
        self.assertEqual(len(retry), 1)
        self.assertEqual(cc.read_json(self.paths.job(retry[0]).job_json)["fingerprint_decision"], "retry")

    def test_non_mr_objects_are_acknowledged_but_not_stored(self):
        self._send(self.listener, modality="SR", count_per_series=1)
        self.assertFalse((self.paths.studies / self.study).exists())
        self.assertEqual(self._dispatch_after_quiet(self.listener), [])

    def test_invalid_uid_is_rejected(self):
        ds = utils.make_mr(self.study, self.series[0])
        ds.SeriesInstanceUID = "../../escape"
        self.assertEqual(self.listener.handle_store(store_event(ds, fake_assoc())), 0xC000)
        self.assertEqual(list(self.paths.studies.glob("**/*.dcm")), [])

    def test_failed_push_keeps_study_pending_and_releases_fingerprint(self):
        broker = FlakyBroker()
        listener = self._make_listener(broker)
        self._send(listener)
        self.assertEqual(self._dispatch_after_quiet(listener), [])
        self.assertTrue((self.paths.studies / self.study / listener_mod.PENDING_MARKER).exists())
        self.assertEqual(list(self.paths.jobs.iterdir()), [])
        jobs = listener.dispatch_once()
        self.assertEqual(len(jobs), 1, "retry must not be mistaken for a duplicate")
        self.assertEqual(broker.length(cc.JOBS), 1)

    def test_pending_study_is_recovered_after_restart(self):
        self._send(self.listener)
        restarted = self._make_listener(self.broker)
        restarted.recover()
        self.clock.now += QUIET - 1
        self.assertEqual(restarted.dispatch_once(), [])
        self.assertEqual(len(self._dispatch_after_quiet(restarted)), 1)

    def test_janitor_keeps_pending_and_purges_idle_dispatched_studies(self):
        self._send(self.listener)
        self.assertEqual(self.listener.janitor(), 0, "pending studies are never purged")
        self._dispatch_after_quiet(self.listener)
        config = listener_mod.ListenerConfig(quiet_seconds=QUIET, min_free_gb=0, study_retention_hours=0)
        janitor = listener_mod.StudyListener(config, self.paths, self.broker, logging.getLogger("t"), clock=self.clock)
        time.sleep(0.01)
        self.assertEqual(janitor.janitor(), 1)
        self.assertFalse((self.paths.studies / self.study).exists())


@unittest.skipUnless(utils.has_module("pynetdicom"), "pynetdicom not installed")
class ListenerNetworkTests(unittest.TestCase):
    """Real DICOM association against the SCP (runs inside the listener image)."""

    def test_c_store_over_the_network(self):
        from pynetdicom import AE
        from pynetdicom.sop_class import MRImageStorage, Verification

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            paths = cc.DataPaths(Path(tmp))
            paths.ensure()
            clock = FakeClock()
            config = listener_mod.ListenerConfig(port=port, quiet_seconds=QUIET, min_free_gb=0)
            listener = listener_mod.StudyListener(config, paths, cc.InMemoryBroker(), logging.getLogger("t"), clock=clock)
            listener.serve(block=False, host="127.0.0.1")
            try:
                study, series = generate_uid(), generate_uid()
                files = [utils.write(utils.make_mr(study, series, instance_number=i), Path(tmp) / f"src{i}.dcm")
                         for i in (1, 2)]
                scu = AE(ae_title="CARPL_TEST")
                scu.add_requested_context(Verification)
                scu.add_requested_context(MRImageStorage, "1.2.840.10008.1.2.1")
                assoc = scu.associate("127.0.0.1", port, ae_title="PROSTATE_AI")
                self.assertTrue(assoc.is_established)
                self.assertEqual(assoc.send_c_echo().Status, 0)
                for path in files:
                    self.assertEqual(assoc.send_c_store(path).Status, 0)
                assoc.release()

                wrong = scu.associate("127.0.0.1", port, ae_title="SOMEONE_ELSE")
                self.assertFalse(wrong.is_established, "called AE title must match")

                deadline = time.time() + 5
                while listener.tracker.due_studies(0) == [] and time.time() < deadline:
                    time.sleep(0.05)
                clock.now += QUIET + 1
                jobs = listener.dispatch_once()
                self.assertEqual(len(jobs), 1)
                stored = sorted((paths.job(jobs[0]).input / series).glob("*.dcm"))
                self.assertEqual(len(stored), 2)
                parity = utils.load_module("carpl_parity", "carpl/tools/parity_check.py")
                for source in files:
                    sop_uid = pydicom.dcmread(source, stop_before_pixels=True).SOPInstanceUID
                    target = paths.studies / study / series / f"{sop_uid}.dcm"
                    self.assertEqual(parity.dataset_bytes(source), parity.dataset_bytes(target),
                                     "dataset bytes must be stored exactly as sent")
            finally:
                listener.shutdown()


if __name__ == "__main__":
    unittest.main()
