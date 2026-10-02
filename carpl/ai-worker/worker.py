#!/usr/bin/env python3
"""CARPL ai-worker: runs the unchanged prostate MRI pipeline for each study.

The worker image is built ``FROM`` the validated inference image
(``prostate-mri-seg:<version>``).  For every queued study it runs exactly the
same command as that image's ENTRYPOINT::

    python /app/prostate_mri_lesion_seg_app -i <job>/input -o <job>/output -m /models

Each study runs in its own subprocess, like the fresh-container-per-case
batch runner, so a crash stays isolated and GPU memory is released between
studies.

The app can exit 0 even when it produced nothing, so the outcome is judged
by the existing ``scripts/case_status.py`` (the same audit schema as HIVE
batches).  Only then are the two canonical outputs queued for return to
CARPL, byte-for-byte unmodified.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.environ.get("CARPL_LIB_DIR", "/opt/carpl/lib"))

import case_status  # noqa: E402  (copied unchanged from scripts/case_status.py)
from carpl_common import (  # noqa: E402
    COMPREHENSIVE_SR_SOP_CLASS,
    EXPLICIT_VR_LITTLE_ENDIAN,
    IMPLICIT_VR_LITTLE_ENDIAN,
    JOBS,
    JOBS_FAILED,
    JOBS_PROCESSING,
    RESULT_FILES,
    RTSTRUCT_SOP_CLASS,
    SEND,
    DataPaths,
    JobPaths,
    append_event,
    atomic_write_json,
    connect_redis,
    data_paths_from_env,
    env_bool,
    env_float,
    env_int,
    env_str,
    fingerprint_key,
    job_state,
    read_json,
    setup_logging,
    sha256_file,
    study_latest_key,
    touch_heartbeat,
    update_job_state,
)

SERVICE = "ai-worker"

# Same contract as scripts/load_and_run.py and scripts/batch_load_and_run.py.
EXPECTED_MODELS = (
    "organ/model.ts",
    "fold0/model_best_fold0.pth.tar",
    "fold1/model_best_fold1.pth.tar",
    "fold2/model_best_fold2.pth.tar",
    "fold3/model_best_fold3.pth.tar",
    "fold4/model_best_fold4.pth.tar",
    "classifier/model_best.pth.tar",
)

SUCCESS_STATUSES = (case_status.Status.SUCCESS_COMPLETE, case_status.Status.SUCCESS_EMPTY_LESION)
FINAL_STATES = frozenset({"delivered", "held", "superseded", "duplicate", "failed", "send_failed"})
RULE_ROLES = {"Rules_T2": "t2", "Rules_ADC": "adc", "Rules_HIGHB": "highb"}
_JOB_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


class ShutdownRequested(Exception):
    """Raised when SIGTERM arrives while a pipeline subprocess is running."""


@dataclass(frozen=True)
class WorkerConfig:
    app_dir: Path = Path("/app/prostate_mri_lesion_seg_app")
    app_workdir: Path = Path("/app")
    models_dir: Path = Path("/models")
    model_manifest: Path = Path("/config/deployment_manifest.json")
    require_model_manifest: bool = False
    require_gpu: bool = True
    min_gpu_memory_gb: float = 11.5
    job_timeout_seconds: float = 900.0
    max_job_attempts: int = 3
    ambiguous_selection_policy: str = "send"
    send_empty_results: bool = True
    delivered_retention_hours: float = 24.0
    failed_retention_days: float = 7.0
    dedup_ttl_seconds: int = 86400
    python_executable: str = sys.executable

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        policy = env_str("AMBIGUOUS_SELECTION_POLICY", "send").lower()
        if policy not in {"send", "hold"}:
            raise ValueError("AMBIGUOUS_SELECTION_POLICY must be 'send' or 'hold'")
        return cls(
            app_dir=Path(env_str("APP_DIR", "/app/prostate_mri_lesion_seg_app")),
            app_workdir=Path(env_str("APP_WORKDIR", "/app")),
            models_dir=Path(env_str("MODELS_DIR", "/models")),
            model_manifest=Path(env_str("MODEL_MANIFEST", "/config/deployment_manifest.json")),
            require_model_manifest=env_bool("REQUIRE_MODEL_MANIFEST", False),
            require_gpu=env_bool("REQUIRE_GPU", True),
            min_gpu_memory_gb=env_float("MIN_GPU_MEMORY_GB", 11.5),
            job_timeout_seconds=env_float("JOB_TIMEOUT_SECONDS", 900.0),
            max_job_attempts=env_int("MAX_JOB_ATTEMPTS", 3),
            ambiguous_selection_policy=policy,
            send_empty_results=env_bool("SEND_EMPTY_RESULTS", True),
            delivered_retention_hours=env_float("DELIVERED_RETENTION_HOURS", 24.0),
            failed_retention_days=env_float("FAILED_RETENTION_DAYS", 7.0),
            dedup_ttl_seconds=env_int("DEDUP_TTL_SECONDS", 86400),
        )


# --------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------- #
def check_models(models_dir: Path) -> "list[str]":
    """Return problems with the mounted models tree (empty list = OK).

    The MONAI Deploy ``NamedModel`` loader requires the root to contain only
    sub-folders holding exactly one file each; any extra file (for example a
    ``deployment_manifest.json`` or ``.DS_Store``) silently changes how the
    organ TorchScript model is discovered.  We therefore demand the exact
    seven-file layout.
    """
    problems: "list[str]" = []
    if not models_dir.is_dir():
        return [f"models directory {models_dir} does not exist"]
    expected = {Path(p) for p in EXPECTED_MODELS}
    expected_dirs = {p.parts[0] for p in expected}
    for rel in sorted(expected):
        if not (models_dir / rel).is_file():
            problems.append(f"missing model file {rel}")
    for entry in sorted(models_dir.iterdir()):
        if entry.name not in expected_dirs:
            problems.append(f"unexpected entry in models root: {entry.name}")
        elif entry.is_dir():
            for child in sorted(entry.iterdir()):
                if Path(entry.name, child.name) not in expected:
                    problems.append(f"unexpected entry in models/{entry.name}: {child.name}")
        else:
            problems.append(f"models/{entry.name} must be a directory")
    return problems


def verify_manifest(models_dir: Path, manifest_path: Path, required: bool) -> "list[str]":
    """Verify SHA-256 checksums against a ``deployment_manifest.json``.

    The manifest format is the one written by ``scripts/package_models.py``
    (``{"organ/model.ts": {"sha256": ..., "bytes": ...}, ...}``).  It must be
    mounted outside ``/models`` (see :func:`check_models`).
    """
    if not manifest_path.is_file():
        return [f"model manifest {manifest_path} not found"] if required else []
    manifest = read_json(manifest_path, default=None)
    if not isinstance(manifest, dict):
        return [f"model manifest {manifest_path} is not valid JSON"]
    problems = []
    for rel in EXPECTED_MODELS:
        entry = manifest.get(rel)
        if not isinstance(entry, dict) or "sha256" not in entry:
            problems.append(f"manifest has no sha256 for {rel}")
            continue
        path = models_dir / rel
        if not path.is_file():
            continue  # already reported by check_models
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(f"checksum mismatch for {rel}: expected {entry['sha256']}, got {actual}")
    return problems


def check_gpu(python_executable: str) -> dict:
    """Probe CUDA in a subprocess so the worker itself never holds GPU memory."""
    probe = (
        "import json, torch\n"
        "info = {'available': torch.cuda.is_available(), 'torch': torch.__version__,"
        " 'cuda': torch.version.cuda}\n"
        "if info['available']:\n"
        "    p = torch.cuda.get_device_properties(0)\n"
        "    info.update(name=p.name, total_memory_gb=round(p.total_memory / 1024**3, 2),"
        " device_count=torch.cuda.device_count())\n"
        "print(json.dumps(info))\n"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe], capture_output=True, text=True, timeout=180
        )
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"available": False, "error": str(exc)}


# --------------------------------------------------------------------- #
# Series-selection traceability (read-only use of the app's own rules)
# --------------------------------------------------------------------- #
def load_selection_rules(app_py: Path) -> "dict[str, dict]":
    """Read the T2/ADC/HIGHB rule JSON strings from ``app.py`` without importing it."""
    tree = ast.parse(app_py.read_text(encoding="utf-8"))
    rules: "dict[str, dict]" = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in RULE_ROLES and isinstance(node.value.value, str):
                parsed = json.loads(node.value.value)
                rules[RULE_ROLES[target.id]] = parsed["selections"][0]["conditions"]
    return rules


def sdk_matches(conditions: dict, attributes: dict) -> bool:
    """String-attribute matching as in MONAI Deploy SDK 2.0 ``_select_series``."""
    for key, value_to_match in conditions.items():
        if not value_to_match:
            continue
        attr_value = attributes.get(key)
        if not attr_value:
            return False
        if isinstance(attr_value, str) and isinstance(value_to_match, str):
            if attr_value.casefold() == value_to_match.casefold():
                continue
            if re.search(value_to_match, attr_value, re.IGNORECASE):
                continue
            return False
        if attr_value != value_to_match:
            return False
    return True


def first_dicom_header(directory: Path):
    import pydicom

    for path in sorted(directory.glob("*.dcm")):
        try:
            return pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
    return None


def selection_record(job: JobPaths, manifest: dict, rules: "dict[str, dict]") -> dict:
    """Candidate series per role (SDK semantics) and the series actually used."""
    candidates: "dict[str, list[str]]" = {role: [] for role in rules}
    for series in manifest.get("series", []):
        attributes = {
            "Modality": series.get("modality", ""),
            "SeriesDescription": series.get("series_description", ""),
        }
        for role, conditions in rules.items():
            if sdk_matches(conditions, attributes):
                candidates[role].append(series.get("series_instance_uid"))
    selected: "dict[str, Optional[str]]" = {}
    for role in rules:
        header = first_dicom_header(job.output / "dicom" / role)
        selected[role] = str(header.SeriesInstanceUID) if header is not None else None
    ambiguous = sorted(role for role, uids in candidates.items() if len(uids) > 1)
    return {"candidates": candidates, "selected": selected, "ambiguous_roles": ambiguous}


# --------------------------------------------------------------------- #
# Send decision
# --------------------------------------------------------------------- #
EXPECTED_RESULT_SOP_CLASSES = {
    "combined_organ_lesion_RTSTRUCT.dcm": RTSTRUCT_SOP_CLASS,
    "prostate_measurements_SR.dcm": COMPREHENSIVE_SR_SOP_CLASS,
}


def decide_send(
    job: JobPaths,
    manifest: dict,
    case: "case_status.CaseStatus",
    selection: dict,
    config: WorkerConfig,
) -> "tuple[str, list[str]]":
    """Return ``("send" | "hold" | "fail", reasons)`` for a finished job."""
    import pydicom

    if case.status not in SUCCESS_STATUSES:
        return "fail", [f"{case.status}: {case.reason}"]
    if case.status == case_status.Status.SUCCESS_EMPTY_LESION and not config.send_empty_results:
        return "hold", ["empty organ result and SEND_EMPTY_RESULTS=false"]

    problems: "list[str]" = []
    study_uid = manifest.get("study_instance_uid")
    input_series = {s.get("series_instance_uid") for s in manifest.get("series", [])}
    t2_header = first_dicom_header(job.output / "dicom" / "t2")
    if t2_header is None:
        problems.append("selected T2 copy missing under output/dicom/t2")

    rtstruct = job.output / RESULT_FILES[0]
    if rtstruct.is_file() and not case_status.is_current_combined_rtstruct(rtstruct):
        problems.append("combined RTSTRUCT does not match the accepted AI_PROSTATE profile")

    seen_sops = set()
    for name in RESULT_FILES:
        path = job.output / name
        if not path.is_file():
            problems.append(f"{name} missing")
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as exc:
            problems.append(f"{name} unreadable: {exc}")
            continue
        if str(ds.SOPClassUID) != EXPECTED_RESULT_SOP_CLASSES[name]:
            problems.append(f"{name} has unexpected SOP class {ds.SOPClassUID}")
        transfer_syntax = str(ds.file_meta.get("TransferSyntaxUID", ""))
        if transfer_syntax not in (EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN):
            problems.append(f"{name} has unsupported transfer syntax {transfer_syntax}")
        if str(ds.get("StudyInstanceUID", "")) != study_uid:
            problems.append(f"{name} StudyInstanceUID does not match the received study")
        if str(ds.get("SeriesInstanceUID", "")) in input_series:
            problems.append(f"{name} reuses an input SeriesInstanceUID")
        if t2_header is not None and str(ds.get("PatientID", "")) != str(t2_header.get("PatientID", "")):
            problems.append(f"{name} PatientID does not match the selected T2 series")
        sop = str(ds.get("SOPInstanceUID", ""))
        if not sop or sop in seen_sops:
            problems.append(f"{name} has a missing or duplicate SOPInstanceUID")
        seen_sops.add(sop)
    if problems:
        return "fail", problems

    if selection.get("ambiguous_roles") and config.ambiguous_selection_policy == "hold":
        return "hold", [
            "more than one candidate series for " + ", ".join(selection["ambiguous_roles"])
            + " and AMBIGUOUS_SELECTION_POLICY=hold"
        ]
    return "send", []


# --------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------- #
class Worker:
    def __init__(self, config: WorkerConfig, paths: DataPaths, broker, logger: logging.Logger) -> None:
        self.config = config
        self.paths = paths
        self.broker = broker
        self.log = logger
        self.rules = load_selection_rules(config.app_dir / "app.py")
        self._stop = False
        self._last_janitor = 0.0

    # -- lifecycle ------------------------------------------------------ #
    def request_stop(self, *_args) -> None:
        self.log.info("Stop requested")
        self._stop = True

    def preflight(self) -> "list[str]":
        problems = check_models(self.config.models_dir)
        problems += verify_manifest(
            self.config.models_dir, self.config.model_manifest, self.config.require_model_manifest
        )
        if not (self.config.app_dir / "__main__.py").is_file():
            problems.append(f"inference app not found at {self.config.app_dir}")
        if set(self.rules) != set(RULE_ROLES.values()):
            problems.append(f"could not read selection rules from {self.config.app_dir / 'app.py'}")
        gpu = check_gpu(self.config.python_executable)
        self.log.info("GPU probe: %s", gpu)
        if self.config.require_gpu:
            if not gpu.get("available"):
                problems.append(f"REQUIRE_GPU=true but CUDA is not available ({gpu.get('error', 'no device')})")
            elif float(gpu.get("total_memory_gb", 0)) < self.config.min_gpu_memory_gb:
                problems.append(
                    f"GPU memory {gpu.get('total_memory_gb')} GB is below MIN_GPU_MEMORY_GB={self.config.min_gpu_memory_gb}"
                )
        append_event(self.paths, SERVICE, "preflight", ok=not problems, problems=problems, gpu=gpu)
        return problems

    def recover(self) -> None:
        moved = self.broker.requeue_processing(JOBS_PROCESSING, JOBS)
        if moved:
            self.log.warning("Re-queued %d job(s) interrupted by a previous shutdown", moved)

    def run_forever(self) -> None:
        while not self._stop:
            touch_heartbeat(self.paths, SERVICE)
            self._maybe_janitor()
            item = self.broker.reserve(JOBS, JOBS_PROCESSING, timeout=5)
            if item is None:
                continue
            raw, payload = item
            try:
                self.process(payload)
            except ShutdownRequested:
                self.log.warning("Shutdown during job %s; it stays queued for recovery", payload.get("job_id"))
                return
            except Exception:
                self.log.exception("Job %s failed unexpectedly", payload.get("job_id"))
                self._fail(payload, "unexpected worker error (see container log)")
            self.broker.ack(JOBS_PROCESSING, raw)

    # -- job processing ------------------------------------------------- #
    def _set_fp_state(self, fingerprint: Optional[str], state: str, job_id: str) -> None:
        if not fingerprint:
            return
        self.broker.set(
            fingerprint_key(fingerprint),
            json.dumps({"state": state, "job_id": job_id}),
            self.config.dedup_ttl_seconds,
        )

    def _fail(self, payload: dict, reason: str) -> None:
        job_id = str(payload.get("job_id", ""))
        if _JOB_ID_RE.match(job_id):
            job = self.paths.job(job_id)
            if job.root.is_dir():
                update_job_state(job, "failed", reason=reason)
        self._set_fp_state(payload.get("fingerprint"), "failed", job_id)
        self.broker.push(JOBS_FAILED, dict(payload, reason=reason))
        append_event(self.paths, SERVICE, "job_failed", job_id=job_id, reason=reason)

    def process(self, payload: dict) -> None:
        job_id = str(payload.get("job_id", ""))
        if not _JOB_ID_RE.match(job_id):
            self.log.error("Discarding queue item with invalid job id: %r", job_id)
            return
        job = self.paths.job(job_id)
        manifest = read_json(job.job_json, default=None)
        if not isinstance(manifest, dict):
            self._fail(payload, "job.json missing or unreadable")
            return
        fingerprint = manifest.get("fingerprint")
        study_uid = manifest.get("study_instance_uid")
        state = job_state(job)

        if state in FINAL_STATES or state == "send_queued":
            self.log.info("Job %s already %s; nothing to do", job_id, state)
            return
        if state == "inferred":
            # Recovered after inference: finish the decision without re-running,
            # because a re-run would mint new SOP UIDs and send CARPL a second
            # result set.
            self.log.info("Job %s recovered after inference; finalising", job_id)
            self._finalise(job, manifest, append_audit=False)
            return

        attempts = int((read_json(job.state_json, default={}) or {}).get("attempts", 0)) + 1
        if attempts > self.config.max_job_attempts:
            self._fail(payload, f"gave up after {self.config.max_job_attempts} attempts")
            return

        latest = self.broker.get(study_latest_key(study_uid)) if study_uid else None
        if latest and latest != job_id:
            update_job_state(job, "superseded", superseded_by=latest)
            self._set_fp_state(fingerprint, "superseded", job_id)
            self.log.info("Job %s superseded by %s before inference; skipped", job_id, latest)
            append_event(self.paths, SERVICE, "job_superseded", job_id=job_id, superseded_by=latest, stage="before_inference")
            return

        update_job_state(job, "running", attempts=attempts)
        self._set_fp_state(fingerprint, "running", job_id)
        self.log.info("Job %s: running pipeline for study %s (attempt %d)", job_id, study_uid, attempts)
        append_event(self.paths, SERVICE, "job_started", job_id=job_id, study_uid=study_uid, attempt=attempts)

        exit_code, elapsed, timed_out = self.run_pipeline(job)
        case = case_status.inspect_case(
            job.output,
            patient_id=job_id,
            input_dir=job.input,
            exit_code=exit_code,
            elapsed_seconds=elapsed,
        )
        if timed_out:
            case.status = case_status.Status.FAIL_RUNTIME
            case.reason = f"Pipeline exceeded JOB_TIMEOUT_SECONDS={self.config.job_timeout_seconds:.0f}. " + case.reason
        case_status.append_csv_row(self.paths.audit_csv, case)
        update_job_state(
            job, "inferred",
            status=case.status, reason=case.reason, exit_code=exit_code, elapsed_seconds=round(elapsed, 2),
        )
        self._set_fp_state(fingerprint, "inferred", job_id)
        self._finalise(job, manifest, append_audit=False, case=case)

    def _finalise(
        self,
        job: JobPaths,
        manifest: dict,
        *,
        append_audit: bool,
        case: "Optional[case_status.CaseStatus]" = None,
    ) -> None:
        job_id = job.job_id
        fingerprint = manifest.get("fingerprint")
        study_uid = manifest.get("study_instance_uid")
        if case is None:
            previous = read_json(job.state_json, default={}) or {}
            case = case_status.inspect_case(
                job.output, patient_id=job_id, input_dir=job.input,
                exit_code=previous.get("exit_code"), elapsed_seconds=previous.get("elapsed_seconds"),
            )
            if append_audit:
                case_status.append_csv_row(self.paths.audit_csv, case)

        selection = selection_record(job, manifest, self.rules)
        atomic_write_json(job.root / "selection.json", selection)
        if selection["ambiguous_roles"]:
            self.log.info(
                "Job %s: more than one candidate series for %s; the pipeline used %s",
                job_id, ", ".join(selection["ambiguous_roles"]),
                {role: selection["selected"].get(role) for role in selection["ambiguous_roles"]},
            )

        decision, reasons = decide_send(job, manifest, case, selection, self.config)
        summary = dict(
            job_id=job_id, study_uid=study_uid, status=case.status, decision=decision,
            reasons=reasons, ambiguous_roles=selection["ambiguous_roles"], selected=selection["selected"],
        )

        if decision == "send":
            latest = self.broker.get(study_latest_key(study_uid)) if study_uid else None
            if latest and latest != job_id:
                update_job_state(job, "superseded", superseded_by=latest)
                self._set_fp_state(fingerprint, "superseded", job_id)
                self.log.info("Job %s superseded by %s after inference; results not sent", job_id, latest)
                append_event(self.paths, SERVICE, "job_superseded", stage="after_inference", superseded_by=latest, **summary)
                return
            files = [str(job.output / name) for name in RESULT_FILES]
            self.broker.push(SEND, {"job_id": job_id, "study_uid": study_uid, "fingerprint": fingerprint, "files": files})
            update_job_state(job, "send_queued", status=case.status)
            self._set_fp_state(fingerprint, "send_queued", job_id)
            self.log.info("Job %s: %s; results queued for CARPL", job_id, case.status)
            append_event(self.paths, SERVICE, "results_queued", **summary)
        elif decision == "hold":
            update_job_state(job, "held", status=case.status, reason="; ".join(reasons))
            self._set_fp_state(fingerprint, "held", job_id)
            self.log.warning("Job %s held for review: %s", job_id, "; ".join(reasons))
            append_event(self.paths, SERVICE, "job_held", **summary)
        else:
            update_job_state(job, "failed", status=case.status, reason="; ".join(reasons))
            self._set_fp_state(fingerprint, "failed", job_id)
            self.log.error("Job %s: nothing sent to CARPL: %s", job_id, "; ".join(reasons))
            append_event(self.paths, SERVICE, "job_failed", **summary)

    def run_pipeline(self, job: JobPaths) -> "tuple[Optional[int], float, bool]":
        """Run the unchanged inference app; returns (exit_code, seconds, timed_out)."""
        if job.output.exists():
            shutil.rmtree(job.output)  # partial output from an interrupted attempt
        shutil.rmtree(job.tmp, ignore_errors=True)
        job.output.mkdir(parents=True)
        job.tmp.mkdir(parents=True)

        cmd = [
            self.config.python_executable, str(self.config.app_dir),
            "-i", str(job.input), "-o", str(job.output), "-m", str(self.config.models_dir),
        ]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CARPL_", "REDIS_")) and k != "PYTHONPATH"}
        env.update(TMPDIR=str(job.tmp), HOME=str(job.tmp))

        started = time.monotonic()
        timed_out = False
        with open(job.pipeline_log, "ab") as log_handle:
            log_handle.write(f"$ {' '.join(cmd)}\n".encode())
            log_handle.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(self.config.app_workdir), env=env,
                stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while True:
                try:
                    exit_code = proc.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    touch_heartbeat(self.paths, SERVICE)
                    if self._stop:
                        self._terminate(proc)
                        raise ShutdownRequested()
                    if time.monotonic() - started > self.config.job_timeout_seconds:
                        timed_out = True
                        self.log.error("Job %s exceeded %.0fs; terminating", job.job_id, self.config.job_timeout_seconds)
                        exit_code = self._terminate(proc)
                        break
        shutil.rmtree(job.tmp, ignore_errors=True)
        return exit_code, time.monotonic() - started, timed_out

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> Optional[int]:
        for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                return proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue
        return proc.poll()

    # -- housekeeping --------------------------------------------------- #
    def _maybe_janitor(self) -> None:
        if time.monotonic() - self._last_janitor < 600:
            return
        self._last_janitor = time.monotonic()
        try:
            self.janitor()
        except Exception:
            self.log.exception("Job janitor failed")

    def janitor(self, now: Optional[float] = None) -> int:
        """Purge finished job folders (they contain copies of patient DICOM)."""
        now = time.time() if now is None else now
        removed = 0
        if not self.paths.jobs.exists():
            return 0
        for job_dir in self.paths.jobs.iterdir():
            if not (job_dir.is_dir() and _JOB_ID_RE.match(job_dir.name)):
                continue
            job = JobPaths(job_dir)
            state = job_state(job)
            marker = job.state_json if job.state_json.exists() else job_dir
            age = now - marker.stat().st_mtime
            if state == "delivered":
                limit = self.config.delivered_retention_hours * 3600
            elif state in FINAL_STATES or state is None:
                limit = self.config.failed_retention_days * 86400
            else:
                continue  # queued / running / inferred / send_queued
            if age >= limit:
                shutil.rmtree(job_dir, ignore_errors=True)
                removed += 1
                append_event(self.paths, SERVICE, "job_purged", job_id=job_dir.name, state=state)
        return removed


def main() -> int:
    log = setup_logging(SERVICE)
    config = WorkerConfig.from_env()
    paths = data_paths_from_env()
    paths.ensure()
    broker = connect_redis(log)
    worker = Worker(config, paths, broker, log)
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)

    problems = worker.preflight()
    if problems:
        for problem in problems:
            log.error("Preflight: %s", problem)
        log.error("Refusing to process jobs until the problems above are fixed")
        time.sleep(60)  # avoid a tight restart loop under restart: unless-stopped
        return 1
    log.info("Preflight passed; models at %s, app at %s", config.models_dir, config.app_dir)
    worker.recover()
    worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
