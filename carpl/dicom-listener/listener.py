#!/usr/bin/env python3
"""CARPL-facing DICOM Storage SCP for the prostate MRI pipeline.

This adapts Hazel's CXR-BMD ``dicom-listener`` (pynetdicom SCP + Redis) to
multi-series MR studies, following CARPL DICOM Push Integration Guide §4.
It handles four things her per-instance design did not need:

* **Study completeness.** Instances accumulate per study.  A job starts only
  after the study has had no open association and no new instance for
  ``STUDY_QUIET_SECONDS``, never on the first image.
* **Pipeline input layout.** Files are stored as
  ``<Study>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm``.  The inference app
  needs one series per folder and ``.dcm`` names, and each job holds
  exactly one study.
* **Deduplication and retransmission.** A duplicate SOP overwrites the same
  file.  An identical resend of an already queued or delivered instance set
  is skipped.  Late or extra instances create a new job over the full set,
  which supersedes the older one.
* **Timeouts for partial studies.** Network, ACSE and DIMSE timeouts abort
  hung associations, and the quiet period then applies.

Bytes are stored exactly as received (``event.encoded_dataset()``).  Nothing
is decoded or re-encoded.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.environ.get("CARPL_LIB_DIR", "/opt/carpl/lib"))

from carpl_common import (  # noqa: E402
    EXPLICIT_VR_LITTLE_ENDIAN,
    FP_SKIP_STATES,
    IMPLICIT_VR_LITTLE_ENDIAN,
    JOBS,
    DataPaths,
    append_event,
    atomic_write_json,
    connect_redis,
    data_paths_from_env,
    env_bool,
    env_float,
    env_int,
    env_list,
    env_str,
    fingerprint_key,
    free_bytes,
    is_valid_uid,
    new_job_id,
    setup_logging,
    study_fingerprint,
    study_latest_key,
    touch_heartbeat,
    update_job_state,
    utc_now_iso,
    validate_ae_title,
    validate_uid,
)

SERVICE = "dicom-listener"
HEALTHCHECK_AE = "PROSTATE_HC"
PENDING_MARKER = ".pending"
DISPATCHED_MARKER = ".dispatched"

# DICOM status codes (PS3.4 Annex B / C)
STATUS_SUCCESS = 0x0000
STATUS_OUT_OF_RESOURCES = 0xA700
STATUS_CANNOT_UNDERSTAND = 0xC000


@dataclass(frozen=True)
class ListenerConfig:
    ae_title: str = "PROSTATE_AI"
    port: int = 11112
    allowed_calling_aes: "tuple[str, ...]" = ()
    require_called_aet: bool = True
    accept_modalities: "frozenset[str]" = frozenset({"MR"})
    quiet_seconds: float = 60.0
    dispatch_interval: float = 5.0
    min_free_gb: float = 20.0
    dedup_ttl_seconds: int = 86400
    study_retention_hours: float = 6.0
    network_timeout: float = 120.0
    acse_timeout: float = 30.0
    dimse_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "ListenerConfig":
        allowed = tuple(
            validate_ae_title(ae, name="ALLOWED_CALLING_AES entry")
            for ae in env_list("ALLOWED_CALLING_AES")
        )
        modalities = frozenset(m.upper() for m in env_list("ACCEPT_MODALITIES")) or frozenset({"MR"})
        if "*" in modalities:
            modalities = frozenset()
        return cls(
            ae_title=validate_ae_title(env_str("DICOM_AE", "PROSTATE_AI"), name="DICOM_AE"),
            port=env_int("DICOM_PORT", 11112),
            allowed_calling_aes=allowed,
            require_called_aet=env_bool("REQUIRE_CALLED_AET", True),
            accept_modalities=modalities,
            quiet_seconds=env_float("STUDY_QUIET_SECONDS", 60.0),
            dispatch_interval=env_float("DISPATCH_INTERVAL_SECONDS", 5.0),
            min_free_gb=env_float("MIN_FREE_GB", 20.0),
            dedup_ttl_seconds=env_int("DEDUP_TTL_SECONDS", 86400),
            study_retention_hours=env_float("STUDY_RETENTION_HOURS", 6.0),
            network_timeout=env_float("DICOM_NETWORK_TIMEOUT", 120.0),
            acse_timeout=env_float("DICOM_ACSE_TIMEOUT", 30.0),
            dimse_timeout=env_float("DICOM_DIMSE_TIMEOUT", 120.0),
        )


@dataclass
class _StudyState:
    last_activity: float
    pending_instances: int = 0
    open_associations: "set[int]" = field(default_factory=set)
    calling_aes: "set[str]" = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


class StudyTracker:
    """In-memory record of each study's receive activity.

    The listener is a single process, so this state stays in memory rather
    than in Redis.  Marker files in each study folder rebuild it after a
    restart.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._studies: "dict[str, _StudyState]" = {}
        self._guard = threading.Lock()

    def study_lock(self, study_uid: str) -> threading.Lock:
        with self._guard:
            state = self._studies.setdefault(study_uid, _StudyState(last_activity=self._clock()))
            return state.lock

    def note_instance(self, study_uid: str, assoc_key: int, calling_ae: str) -> int:
        """Record one stored instance; returns the pending count after the update."""
        with self._guard:
            state = self._studies.setdefault(study_uid, _StudyState(last_activity=self._clock()))
            state.last_activity = self._clock()
            state.pending_instances += 1
            state.open_associations.add(assoc_key)
            if calling_ae:
                state.calling_aes.add(calling_ae)
            return state.pending_instances

    def restore_pending(self, study_uid: str, calling_aes: "set[str]") -> None:
        with self._guard:
            state = self._studies.setdefault(study_uid, _StudyState(last_activity=self._clock()))
            state.last_activity = self._clock()
            state.pending_instances = max(state.pending_instances, 1)
            state.calling_aes.update(calling_aes)

    def association_closed(self, assoc_key: int) -> "list[str]":
        touched = []
        with self._guard:
            for study_uid, state in self._studies.items():
                if assoc_key in state.open_associations:
                    state.open_associations.discard(assoc_key)
                    state.last_activity = self._clock()
                    touched.append(study_uid)
        return touched

    def due_studies(self, quiet_seconds: float, now: Optional[float] = None) -> "list[str]":
        now = self._clock() if now is None else now
        with self._guard:
            return sorted(
                uid
                for uid, state in self._studies.items()
                if state.pending_instances > 0
                and not state.open_associations
                and now - state.last_activity >= quiet_seconds
            )

    def take_pending(self, study_uid: str) -> "tuple[int, set[str]]":
        """Reset the pending counter (caller holds the study lock)."""
        with self._guard:
            state = self._studies.get(study_uid)
            if state is None:
                return 0, set()
            count, aes = state.pending_instances, set(state.calling_aes)
            state.pending_instances = 0
            state.calling_aes.clear()
            return count, aes

    def restore_after_failure(self, study_uid: str, count: int, calling_aes: "set[str]") -> None:
        with self._guard:
            state = self._studies.setdefault(study_uid, _StudyState(last_activity=self._clock()))
            state.pending_instances += max(count, 1)
            state.calling_aes.update(calling_aes)

    def forget(self, study_uid: str) -> None:
        with self._guard:
            state = self._studies.get(study_uid)
            if state is not None and state.pending_instances == 0 and not state.open_associations:
                del self._studies[study_uid]

    def pending_count(self, study_uid: str) -> int:
        with self._guard:
            state = self._studies.get(study_uid)
            return state.pending_instances if state else 0


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


_SUMMARY_TAGS = [
    "Modality",
    "SeriesDescription",
    "SeriesNumber",
    "InstanceNumber",
    "SOPClassUID",
]


def summarise_series(series_dir: Path) -> dict:
    """Header-only summary of one stored series, for ``job.json`` traceability."""
    import pydicom  # local import: only needed at dispatch time

    files = sorted(series_dir.glob("*.dcm"))
    summary: dict = {
        "series_instance_uid": series_dir.name,
        "instances": len(files),
    }
    instance_numbers: "list[int]" = []
    for index, path in enumerate(files):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=_SUMMARY_TAGS)
        except Exception as exc:  # recorded, never fatal for dispatch
            summary.setdefault("unreadable_files", []).append(f"{path.name}: {exc}")
            continue
        if index == 0:
            summary["modality"] = str(ds.get("Modality", ""))
            summary["series_description"] = str(ds.get("SeriesDescription", ""))
            summary["series_number"] = str(ds.get("SeriesNumber", ""))
            summary["sop_class_uid"] = str(ds.get("SOPClassUID", ""))
        number = ds.get("InstanceNumber")
        if number not in (None, ""):
            try:
                instance_numbers.append(int(number))
            except (TypeError, ValueError):
                pass
    if instance_numbers:
        unique = sorted(set(instance_numbers))
        expected = unique[-1] - unique[0] + 1
        summary["instance_number_range"] = [unique[0], unique[-1]]
        if len(unique) != expected or len(unique) != len(instance_numbers):
            summary["instance_number_warning"] = (
                f"{len(instance_numbers)} instances, {len(unique)} unique InstanceNumbers "
                f"spanning {expected}; the series may be incomplete or contain repeats"
            )
    return summary


class StudyListener:
    """Storage SCP, study tracker, dispatcher and janitor."""

    def __init__(
        self,
        config: ListenerConfig,
        paths: DataPaths,
        broker,
        logger: logging.Logger,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.paths = paths
        self.broker = broker
        self.log = logger
        self.clock = clock
        self.tracker = StudyTracker(clock)
        self._server = None
        self._stop = threading.Event()
        self._dropped: "dict[int, int]" = {}
        self._dropped_guard = threading.Lock()
        self._last_janitor = 0.0

    # ------------------------------------------------------------------ #
    # DICOM association handling
    # ------------------------------------------------------------------ #
    def build_ae(self):
        from pynetdicom import AE, AllStoragePresentationContexts
        from pynetdicom.sop_class import Verification

        ae = AE(ae_title=self.config.ae_title)
        ae.network_timeout = self.config.network_timeout
        ae.acse_timeout = self.config.acse_timeout
        ae.dimse_timeout = self.config.dimse_timeout
        ae.maximum_associations = 16
        ae.require_called_aet = self.config.require_called_aet
        if self.config.allowed_calling_aes:
            ae.require_calling_aet = list(self.config.allowed_calling_aes) + [HEALTHCHECK_AE]
        transfer_syntaxes = [EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN]
        ae.add_supported_context(Verification, transfer_syntaxes)
        for context in AllStoragePresentationContexts:
            ae.add_supported_context(context.abstract_syntax, transfer_syntaxes)
        return ae

    def handle_store(self, event) -> int:
        try:
            return self._handle_store(event)
        except Exception:  # never let one instance kill the association thread
            self.log.exception("Unexpected error while storing an instance")
            return STATUS_CANNOT_UNDERSTAND

    def _handle_store(self, event) -> int:
        assoc_key = id(event.assoc)
        calling_ae = str(event.assoc.requestor.ae_title).strip()
        ds = event.dataset
        try:
            study_uid = validate_uid(ds.get("StudyInstanceUID"), name="StudyInstanceUID")
            series_uid = validate_uid(ds.get("SeriesInstanceUID"), name="SeriesInstanceUID")
            sop_uid = validate_uid(ds.get("SOPInstanceUID"), name="SOPInstanceUID")
        except ValueError as exc:
            self.log.warning("Rejected instance from %s: %s", calling_ae, exc)
            append_event(self.paths, SERVICE, "instance_rejected", calling_ae=calling_ae, reason=str(exc))
            return STATUS_CANNOT_UNDERSTAND

        modality = str(ds.get("Modality", "")).strip().upper()
        if self.config.accept_modalities and modality not in self.config.accept_modalities:
            # Accepted but not stored. The pipeline selects MR series only,
            # and this also ignores our own RTSTRUCT/SR if CARPL re-pushes it.
            with self._dropped_guard:
                self._dropped[assoc_key] = self._dropped.get(assoc_key, 0) + 1
            return STATUS_SUCCESS

        if free_bytes(self.paths.root) < self.config.min_free_gb * 1024 ** 3:
            self.log.error("Refusing instance: free space below MIN_FREE_GB=%s", self.config.min_free_gb)
            append_event(self.paths, SERVICE, "instance_refused_disk_full", study_uid=study_uid)
            return STATUS_OUT_OF_RESOURCES

        encoded = event.encoded_dataset()
        self.paths.studies_tmp.mkdir(parents=True, exist_ok=True)
        tmp_path = self.paths.studies_tmp / f"{sop_uid}.{uuid.uuid4().hex}.part"
        with open(tmp_path, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        study_dir = self.paths.studies / study_uid
        series_dir = study_dir / series_uid
        with self.tracker.study_lock(study_uid):
            series_dir.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_path, series_dir / f"{sop_uid}.dcm")
            pending = self.tracker.note_instance(study_uid, assoc_key, calling_ae)
            if pending == 1:
                (study_dir / PENDING_MARKER).write_text(
                    json.dumps({"calling_ae": calling_ae, "first_received": utc_now_iso()}),
                    encoding="utf-8",
                )
                self.log.info("Receiving study %s from %s", study_uid, calling_ae)
        return STATUS_SUCCESS

    def handle_conn_close(self, event) -> None:
        assoc_key = id(event.assoc)
        with self._dropped_guard:
            dropped = self._dropped.pop(assoc_key, 0)
        if dropped:
            self.log.info(
                "Association from %s closed; %d non-%s instance(s) accepted but not stored",
                str(event.assoc.requestor.ae_title).strip(),
                dropped,
                "/".join(sorted(self.config.accept_modalities)),
            )
        self.tracker.association_closed(assoc_key)

    def serve(self, *, block: bool = True, host: str = ""):
        from pynetdicom import evt

        ae = self.build_ae()
        handlers = [
            (evt.EVT_C_STORE, self.handle_store),
            (evt.EVT_CONN_CLOSE, self.handle_conn_close),
        ]
        self.log.info(
            "Storage SCP %s listening on port %d (quiet period %.0fs, modalities %s)",
            self.config.ae_title,
            self.config.port,
            self.config.quiet_seconds,
            ",".join(sorted(self.config.accept_modalities)) or "ALL",
        )
        self._server = ae.start_server((host, self.config.port), block=block, evt_handlers=handlers)
        return self._server

    def shutdown(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()

    # ------------------------------------------------------------------ #
    # Restart recovery
    # ------------------------------------------------------------------ #
    def recover(self) -> None:
        """Remove partial temp files and restore timers for undispatched studies."""
        if self.paths.studies_tmp.exists():
            for leftover in self.paths.studies_tmp.iterdir():
                leftover.unlink(missing_ok=True)
        if not self.paths.studies.exists():
            return
        for study_dir in self.paths.studies.iterdir():
            marker = study_dir / PENDING_MARKER
            if study_dir.is_dir() and is_valid_uid(study_dir.name) and marker.exists():
                try:
                    calling_ae = json.loads(marker.read_text(encoding="utf-8")).get("calling_ae", "")
                except (OSError, ValueError):
                    calling_ae = ""
                self.tracker.restore_pending(study_dir.name, {calling_ae} if calling_ae else set())
                self.log.info("Recovered pending study %s; it will dispatch after the quiet period", study_dir.name)

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def dispatch_once(self, now: Optional[float] = None) -> "list[str]":
        """Dispatch every study that is due; returns the created job ids."""
        created = []
        for study_uid in self.tracker.due_studies(self.config.quiet_seconds, now=now):
            job_id = self.snapshot_study(study_uid)
            if job_id:
                created.append(job_id)
        return created

    def snapshot_study(self, study_uid: str) -> Optional[str]:
        study_dir = self.paths.studies / study_uid
        job_id = new_job_id()
        job = self.paths.job(job_id)

        # Only the hardlink snapshot runs under the study lock, so a large
        # study does not block C-STOREs for long.  An instance that arrives
        # after this point counts as pending again and gives a later job.
        with self.tracker.study_lock(study_uid):
            count, calling_aes = self.tracker.take_pending(study_uid)
            if count == 0:
                return None
            try:
                sop_uids, series = self._link_study(study_dir, job.input)
            except Exception:
                self.log.exception("Snapshot of study %s failed; will retry", study_uid)
                shutil.rmtree(job.root, ignore_errors=True)
                self.tracker.restore_after_failure(study_uid, count, calling_aes)
                return None
            (study_dir / PENDING_MARKER).unlink(missing_ok=True)
            self._mark_dispatched(study_dir)

        if not sop_uids:
            self.log.warning("Study %s has no stored instances; nothing to dispatch", study_uid)
            shutil.rmtree(job.root, ignore_errors=True)
            return None

        fingerprint = study_fingerprint(sop_uids)
        registered = False
        try:
            decision = self._register_fingerprint(fingerprint, job_id)
            registered = decision != "duplicate"
            if decision == "duplicate":
                shutil.rmtree(job.root, ignore_errors=True)
                self.log.info(
                    "Study %s resent with an identical instance set (%d instances); already handled, skipped",
                    study_uid,
                    len(sop_uids),
                )
                append_event(
                    self.paths, SERVICE, "duplicate_skipped",
                    study_uid=study_uid, fingerprint=fingerprint, instances=len(sop_uids),
                )
                return None

            manifest = {
                "job_id": job_id,
                "study_instance_uid": study_uid,
                "calling_aes": sorted(calling_aes),
                "created_at": utc_now_iso(),
                "fingerprint": fingerprint,
                "fingerprint_decision": decision,
                "instance_count": len(sop_uids),
                "series": [summarise_series(job.input / s) for s in series],
                "quiet_seconds": self.config.quiet_seconds,
            }
            atomic_write_json(job.job_json, manifest)
            update_job_state(job, "queued", fingerprint=fingerprint)
            self.broker.set(study_latest_key(study_uid), job_id, self.config.dedup_ttl_seconds)
            self.broker.push(JOBS, {"job_id": job_id, "study_uid": study_uid, "fingerprint": fingerprint})
        except Exception:
            # Keep the study pending so the next dispatch cycle retries it.
            self.log.exception("Dispatch of study %s failed; will retry", study_uid)
            shutil.rmtree(job.root, ignore_errors=True)
            if registered:
                # Release the dedup claim, otherwise the retry would be
                # mistaken for a duplicate of this failed dispatch.
                try:
                    self.broker.delete(fingerprint_key(fingerprint))
                except Exception:
                    self.log.warning("Could not release fingerprint %s", fingerprint)
            with self.tracker.study_lock(study_uid):
                self.tracker.restore_after_failure(study_uid, count, calling_aes)
                if study_dir.is_dir() and not (study_dir / PENDING_MARKER).exists():
                    (study_dir / PENDING_MARKER).write_text(
                        json.dumps({"calling_ae": ",".join(sorted(calling_aes)), "first_received": utc_now_iso()}),
                        encoding="utf-8",
                    )
            return None

        self.log.info(
            "Queued job %s for study %s: %d instances in %d series",
            job_id, study_uid, len(sop_uids), len(series),
        )
        append_event(
            self.paths, SERVICE, "job_queued",
            job_id=job_id, study_uid=study_uid, fingerprint=fingerprint,
            instances=len(sop_uids), series=len(series), calling_aes=sorted(calling_aes),
        )
        return job_id

    @staticmethod
    def _link_study(study_dir: Path, destination: Path) -> "tuple[list[str], list[str]]":
        sop_uids: "list[str]" = []
        series: "list[str]" = []
        destination.mkdir(parents=True, exist_ok=False)
        if not study_dir.is_dir():
            return sop_uids, series
        for series_dir in sorted(study_dir.iterdir()):
            if not (series_dir.is_dir() and is_valid_uid(series_dir.name)):
                continue
            files = sorted(series_dir.glob("*.dcm"))
            if not files:
                continue
            target = destination / series_dir.name
            target.mkdir()
            for path in files:
                _link_or_copy(path, target / path.name)
                sop_uids.append(path.stem)
            series.append(series_dir.name)
        return sop_uids, series

    def _register_fingerprint(self, fingerprint: str, job_id: str) -> str:
        key = fingerprint_key(fingerprint)
        value = json.dumps({"state": "queued", "job_id": job_id})
        if self.broker.set_nx(key, value, self.config.dedup_ttl_seconds):
            return "new"
        existing = self.broker.get(key)
        try:
            state = json.loads(existing).get("state") if existing else None
        except (TypeError, ValueError):
            state = None
        if state in FP_SKIP_STATES:
            return "duplicate"
        # Previous attempt failed: treat the resend as a retry.
        self.broker.set(key, value, self.config.dedup_ttl_seconds)
        return "retry"

    @staticmethod
    def _mark_dispatched(study_dir: Path) -> None:
        if study_dir.is_dir():
            (study_dir / DISPATCHED_MARKER).write_text(utc_now_iso(), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #
    def janitor(self) -> int:
        """Purge accumulated study stores that are idle beyond the retention period."""
        if not self.paths.studies.exists():
            return 0
        cutoff = time.time() - self.config.study_retention_hours * 3600
        removed = 0
        for study_dir in self.paths.studies.iterdir():
            if not (study_dir.is_dir() and is_valid_uid(study_dir.name)):
                continue
            if (study_dir / PENDING_MARKER).exists() or self.tracker.pending_count(study_dir.name):
                continue
            try:
                newest = max(p.stat().st_mtime for p in study_dir.rglob("*"))
            except ValueError:
                newest = study_dir.stat().st_mtime
            if newest < cutoff:
                with self.tracker.study_lock(study_dir.name):
                    if self.tracker.pending_count(study_dir.name):
                        continue
                    shutil.rmtree(study_dir, ignore_errors=True)
                self.tracker.forget(study_dir.name)
                removed += 1
                append_event(self.paths, SERVICE, "study_store_purged", study_uid=study_dir.name)
        return removed

    def run_dispatcher(self) -> None:
        while not self._stop.is_set():
            touch_heartbeat(self.paths, SERVICE)
            try:
                self.dispatch_once()
                if time.monotonic() - self._last_janitor > 600:
                    self._last_janitor = time.monotonic()
                    self.janitor()
            except Exception:
                self.log.exception("Dispatcher iteration failed")
            self._stop.wait(self.config.dispatch_interval)

    def start_dispatcher(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_dispatcher, name="dispatcher", daemon=True)
        thread.start()
        return thread


def main() -> int:
    log = setup_logging(SERVICE)
    config = ListenerConfig.from_env()
    paths = data_paths_from_env()
    paths.ensure()
    broker = connect_redis(log)
    listener = StudyListener(config, paths, broker, log)
    listener.recover()
    listener.start_dispatcher()
    append_event(paths, SERVICE, "service_started", ae_title=config.ae_title, port=config.port)
    listener.serve(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
