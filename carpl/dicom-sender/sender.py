#!/usr/bin/env python3
"""CARPL dicom-sender: C-STOREs the pipeline results back to CARPL.

This adapts Hazel's CXR-BMD ``dicom-sender`` in three ways:

* The two existing pipeline outputs, ``combined_organ_lesion_RTSTRUCT.dcm``
  and ``prostate_measurements_SR.dcm``, are sent from their file paths
  without modification.  pynetdicom streams the encoded file, so the bytes
  CARPL receives are exactly the bytes the pipeline wrote.
* Each file's presentation context is proposed with its own SOP class and
  transfer syntax, taken from its file meta.
* Failed sends are retried with exponential backoff and then moved to a
  dead-letter list.  Delivered jobs are never sent twice.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.environ.get("CARPL_LIB_DIR", "/opt/carpl/lib"))

from carpl_common import (  # noqa: E402
    EXPLICIT_VR_LITTLE_ENDIAN,
    IMPLICIT_VR_LITTLE_ENDIAN,
    SEND,
    SEND_FAILED,
    SEND_PROCESSING,
    DataPaths,
    append_event,
    connect_redis,
    data_paths_from_env,
    env_bool,
    env_float,
    env_int,
    env_str,
    fingerprint_key,
    job_state,
    setup_logging,
    touch_heartbeat,
    update_job_state,
    validate_ae_title,
)

SERVICE = "dicom-sender"

# C-STORE success and warning statuses (PS3.4 Table B.2-1).  Warnings mean
# the object was stored (possibly with coercion), so they count as delivered.
SUCCESS_STATUSES = {0x0000, 0xB000, 0xB006, 0xB007}


@dataclass(frozen=True)
class SenderConfig:
    calling_ae: str = "PROSTATE_AI"
    dest_ae: str = "CARPL-RECIEVE"
    dest_host: str = "rapier-orthanc-prefect"
    dest_port: int = 4242
    max_attempts: int = 8
    backoff_base: float = 30.0
    backoff_max: float = 900.0
    allow_transcode: bool = False
    dedup_ttl_seconds: int = 86400
    network_timeout: float = 60.0
    acse_timeout: float = 30.0
    dimse_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "SenderConfig":
        return cls(
            calling_ae=validate_ae_title(env_str("CALLING_AE", "PROSTATE_AI"), name="CALLING_AE"),
            dest_ae=validate_ae_title(env_str("DEST_AE", "CARPL-RECIEVE"), name="DEST_AE"),
            dest_host=env_str("DEST_HOST", "rapier-orthanc-prefect"),
            dest_port=env_int("DEST_PORT", 4242),
            max_attempts=env_int("SEND_MAX_ATTEMPTS", 8),
            backoff_base=env_float("SEND_BACKOFF_BASE_SECONDS", 30.0),
            backoff_max=env_float("SEND_BACKOFF_MAX_SECONDS", 900.0),
            allow_transcode=env_bool("SEND_ALLOW_TRANSCODE", False),
            dedup_ttl_seconds=env_int("DEDUP_TTL_SECONDS", 86400),
            network_timeout=env_float("DICOM_NETWORK_TIMEOUT", 60.0),
            acse_timeout=env_float("DICOM_ACSE_TIMEOUT", 30.0),
            dimse_timeout=env_float("DICOM_DIMSE_TIMEOUT", 120.0),
        )


def file_contexts(paths: "list[Path]", allow_transcode: bool) -> "dict[str, list[str]]":
    """Map SOP Class UID -> transfer syntaxes to propose, from each file's meta."""
    import pydicom

    contexts: "dict[str, list[str]]" = {}
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        sop_class = str(ds.SOPClassUID)
        transfer_syntax = str(ds.file_meta.TransferSyntaxUID)
        syntaxes = contexts.setdefault(sop_class, [])
        if transfer_syntax not in syntaxes:
            syntaxes.append(transfer_syntax)
        if allow_transcode:
            for alternative in (EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN):
                if alternative not in syntaxes:
                    syntaxes.append(alternative)
    return contexts


class Sender:
    def __init__(self, config: SenderConfig, paths: DataPaths, broker, logger: logging.Logger) -> None:
        self.config = config
        self.paths = paths
        self.broker = broker
        self.log = logger
        self._stop = False

    def request_stop(self, *_args) -> None:
        self._stop = True

    # -- DICOM ---------------------------------------------------------- #
    def send_files(self, files: "list[Path]") -> "tuple[bool, list[dict]]":
        """Send all files over one association; returns (all_ok, per-file results)."""
        from pynetdicom import AE

        ae = AE(ae_title=self.config.calling_ae)
        ae.network_timeout = self.config.network_timeout
        ae.acse_timeout = self.config.acse_timeout
        ae.dimse_timeout = self.config.dimse_timeout
        for sop_class, syntaxes in file_contexts(files, self.config.allow_transcode).items():
            ae.add_requested_context(sop_class, syntaxes)

        assoc = ae.associate(self.config.dest_host, self.config.dest_port, ae_title=self.config.dest_ae)
        if not assoc.is_established:
            return False, [{"error": (
                f"association with {self.config.dest_ae}@{self.config.dest_host}:{self.config.dest_port} "
                f"was rejected or aborted"
            )}]
        results = []
        try:
            for path in files:
                results.append(self._send_one(assoc, path))
        finally:
            if assoc.is_established:
                assoc.release()
        return all(r.get("status") in SUCCESS_STATUSES for r in results), results

    def _send_one(self, assoc, path: Path) -> dict:
        try:
            status = assoc.send_c_store(path)
        except ValueError as exc:
            # Raised when the accepted transfer syntax differs from the file's.
            if not self.config.allow_transcode:
                return {"file": path.name, "status": None, "error": f"not sent unmodified: {exc}"}
            import pydicom

            self.log.warning("Transcoding %s for the accepted transfer syntax (SEND_ALLOW_TRANSCODE=true)", path.name)
            status = assoc.send_c_store(pydicom.dcmread(path))
        code = int(status.Status) if status and "Status" in status else None
        result = {"file": path.name, "status": code}
        if code is None:
            result["error"] = "no C-STORE response (timeout or aborted association)"
        elif code not in SUCCESS_STATUSES:
            result["error"] = f"C-STORE status 0x{code:04X}"
        elif code != 0x0000:
            result["warning"] = f"C-STORE warning status 0x{code:04X}"
        return result

    # -- queue ---------------------------------------------------------- #
    def recover(self) -> None:
        moved = self.broker.requeue_processing(SEND_PROCESSING, SEND)
        if moved:
            self.log.warning("Re-queued %d send(s) interrupted by a previous shutdown", moved)

    def run_forever(self) -> None:
        while not self._stop:
            touch_heartbeat(self.paths, SERVICE)
            item = self.broker.reserve(SEND, SEND_PROCESSING, timeout=5)
            if item is None:
                continue
            raw, payload = item
            try:
                finished = self.process(payload)
            except Exception:
                self.log.exception("Send for job %s failed unexpectedly", payload.get("job_id"))
                finished = self._give_up(payload, "unexpected sender error (see container log)")
            if finished:
                self.broker.ack(SEND_PROCESSING, raw)

    def _set_fp_state(self, fingerprint: Optional[str], state: str, job_id: str) -> None:
        if fingerprint:
            self.broker.set(
                fingerprint_key(fingerprint),
                json.dumps({"state": state, "job_id": job_id}),
                self.config.dedup_ttl_seconds,
            )

    def _give_up(self, payload: dict, reason: str) -> bool:
        job_id = str(payload.get("job_id", ""))
        job = self.paths.job(job_id)
        if job_id and job.root.is_dir():
            update_job_state(job, "send_failed", reason=reason)
        self._set_fp_state(payload.get("fingerprint"), "send_failed", job_id)
        self.broker.push(SEND_FAILED, dict(payload, reason=reason))
        self.log.error("Job %s: results NOT delivered to CARPL: %s", job_id, reason)
        append_event(self.paths, SERVICE, "send_failed", job_id=job_id, reason=reason)
        return True

    def _resolve_files(self, payload: dict) -> "tuple[Optional[list[Path]], str]":
        job_id = str(payload.get("job_id", ""))
        job = self.paths.job(job_id)
        output_root = job.output.resolve()
        files = []
        for name in payload.get("files", []):
            path = Path(name).resolve()
            if output_root not in path.parents:
                return None, f"refusing to send {name}: outside the job output folder"
            if not path.is_file():
                return None, f"result file missing: {name}"
            files.append(path)
        if not files:
            return None, "no result files listed"
        return files, ""

    def process(self, payload: dict) -> bool:
        """Deliver one job's results.  Returns True when the queue item is finished."""
        job_id = str(payload.get("job_id", ""))
        job = self.paths.job(job_id)
        if job_state(job) == "delivered":
            self.log.info("Job %s already delivered; skipping duplicate send request", job_id)
            return True
        files, problem = self._resolve_files(payload)
        if files is None:
            return self._give_up(payload, problem)

        destination = f"{self.config.dest_ae}@{self.config.dest_host}:{self.config.dest_port}"
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                ok, results = self.send_files(files)
            except Exception as exc:
                ok, results = False, [{"error": f"{type(exc).__name__}: {exc}"}]
            if ok:
                update_job_state(job, "delivered", destination=destination, attempts=attempt, results=results)
                self._set_fp_state(payload.get("fingerprint"), "delivered", job_id)
                self.log.info(
                    "Job %s: delivered %s to %s (attempt %d)",
                    job_id, ", ".join(p.name for p in files), destination, attempt,
                )
                append_event(
                    self.paths, SERVICE, "results_delivered",
                    job_id=job_id, study_uid=payload.get("study_uid"), destination=destination,
                    attempt=attempt, results=results,
                )
                return True
            delay = min(self.config.backoff_max, self.config.backoff_base * (2 ** (attempt - 1)))
            self.log.warning(
                "Job %s: send attempt %d/%d to %s failed: %s",
                job_id, attempt, self.config.max_attempts, destination, results,
            )
            append_event(self.paths, SERVICE, "send_attempt_failed", job_id=job_id, attempt=attempt, results=results)
            if attempt == self.config.max_attempts:
                break
            if not self._sleep(delay):
                return False  # shutting down: leave the item in the processing list
        return self._give_up(payload, f"{self.config.max_attempts} send attempts to {destination} failed")

    def _sleep(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop:
                return False
            touch_heartbeat(self.paths, SERVICE)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return True


def main() -> int:
    log = setup_logging(SERVICE)
    config = SenderConfig.from_env()
    paths = data_paths_from_env()
    paths.ensure()
    broker = connect_redis(log)
    sender = Sender(config, paths, broker, log)
    signal.signal(signal.SIGTERM, sender.request_stop)
    signal.signal(signal.SIGINT, sender.request_stop)
    log.info(
        "Sending results as %s to %s@%s:%s",
        config.calling_ae, config.dest_ae, config.dest_host, config.dest_port,
    )
    append_event(paths, SERVICE, "service_started", destination=f"{config.dest_ae}@{config.dest_host}:{config.dest_port}")
    sender.recover()
    sender.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
