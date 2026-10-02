"""Shared helpers for the prostate MRI CARPL DICOM-push services.

This module is copied into every CARPL image (dicom-listener, ai-worker,
dicom-sender and the local carpl-receiver stub).  It deliberately depends on
the standard library only; Redis is imported lazily by ``connect_redis`` so
that unit tests can use :class:`InMemoryBroker` instead.

Nothing here touches the inference application.  The worker runs the
unchanged MONAI Deploy app as a subprocess and only reads its outputs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

LOG = logging.getLogger("carpl")

# --------------------------------------------------------------------- #
# DICOM constants
# --------------------------------------------------------------------- #
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"   # RT Structure Set Storage
COMPREHENSIVE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.33"
EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
IMPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2"

# The two existing pipeline outputs that are returned to CARPL, unmodified.
RESULT_FILES = (
    "combined_organ_lesion_RTSTRUCT.dcm",
    "prostate_measurements_SR.dcm",
)

AE_TITLE_MAX_LENGTH = 16
UID_MAX_LENGTH = 64
_UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")

# --------------------------------------------------------------------- #
# Redis keys (all prefixed so they cannot collide with other stacks)
# --------------------------------------------------------------------- #
JOBS = "prostate:jobs"
JOBS_PROCESSING = "prostate:jobs:processing"
JOBS_FAILED = "prostate:jobs:failed"
SEND = "prostate:send"
SEND_PROCESSING = "prostate:send:processing"
SEND_FAILED = "prostate:send:failed"


def fingerprint_key(fingerprint: str) -> str:
    return f"prostate:fp:{fingerprint}"


def study_latest_key(study_uid: str) -> str:
    return f"prostate:study:{study_uid}:latest"


# Fingerprint states.  A resend of an identical instance set is skipped while
# its fingerprint is in one of these states; any other state (failed,
# send_failed, superseded) lets the resend act as a retry.
FP_SKIP_STATES = frozenset(
    {"queued", "running", "inferred", "send_queued", "delivered", "held"}
)


# --------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------- #
def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer: {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number: {value!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean: {value!r}")


def env_list(name: str) -> "list[str]":
    value = os.environ.get(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #
def validate_ae_title(value: str, *, name: str = "AE title") -> str:
    """Return a stripped AE title or raise ``ValueError``.

    DICOM AE titles are 1-16 characters of the default character repertoire,
    excluding backslash and control characters.
    """
    stripped = (value or "").strip()
    if not stripped:
        raise ValueError(f"{name} must not be empty")
    if len(stripped) > AE_TITLE_MAX_LENGTH:
        raise ValueError(f"{name} {stripped!r} exceeds {AE_TITLE_MAX_LENGTH} characters")
    if "\\" in stripped or any(ord(ch) < 0x20 or ord(ch) > 0x7E for ch in stripped):
        raise ValueError(f"{name} {stripped!r} contains invalid characters")
    return stripped


def is_valid_uid(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= UID_MAX_LENGTH
        and bool(_UID_RE.match(value))
    )


def validate_uid(value: object, *, name: str = "UID") -> str:
    """Return ``value`` if it is a well-formed DICOM UID.

    UIDs are used as directory and file names, so this also prevents path
    traversal through crafted datasets.
    """
    text = str(value).strip() if value is not None else ""
    if not is_valid_uid(text):
        raise ValueError(f"Invalid {name}: {text!r}")
    return text


# --------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------- #
def utc_now_iso() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def new_job_id() -> str:
    """Sortable, collision-resistant job identifier, e.g. 20260928T101500Z-ab12cd34."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def study_fingerprint(sop_instance_uids: Iterable[str]) -> str:
    """SHA-256 over the sorted, de-duplicated SOP Instance UIDs of a job."""
    joined = "\n".join(sorted(set(sop_instance_uids)))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def sha256_file(path: "os.PathLike[str] | str", chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataPaths:
    """Layout of the single ``/data`` bind mount shared by all services.

    A single mount keeps hardlinks and ``os.replace`` on one filesystem.
    """

    root: Path

    @property
    def studies(self) -> Path:
        return self.root / "studies"

    @property
    def studies_tmp(self) -> Path:
        return self.studies / ".tmp"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    @property
    def heartbeats(self) -> Path:
        return self.root / "heartbeats"

    @property
    def audit_csv(self) -> Path:
        return self.audit / "carpl_audit_log.csv"

    @property
    def events_jsonl(self) -> Path:
        return self.audit / "events.jsonl"

    def job(self, job_id: str) -> "JobPaths":
        return JobPaths(self.jobs / job_id)

    def ensure(self) -> None:
        for directory in (self.studies, self.studies_tmp, self.jobs, self.audit, self.heartbeats):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def job_id(self) -> str:
        return self.root.name

    @property
    def job_json(self) -> Path:
        return self.root / "job.json"

    @property
    def state_json(self) -> Path:
        return self.root / "state.json"

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def pipeline_log(self) -> Path:
        return self.root / "pipeline.log"


def data_paths_from_env() -> DataPaths:
    return DataPaths(Path(env_str("DATA_DIR", "/data")))


def atomic_write_json(path: "os.PathLike[str] | str", payload: Any) -> None:
    """Write JSON so readers never observe a partially written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: "os.PathLike[str] | str", default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def update_job_state(job: JobPaths, state: str, **fields: Any) -> dict:
    """Record a job state transition in ``state.json`` (append-only history)."""
    current = read_json(job.state_json, default={}) or {}
    history = list(current.get("history", []))
    entry = {"state": state, "at": utc_now_iso()}
    entry.update({key: value for key, value in fields.items() if value is not None})
    history.append(entry)
    current.update(fields)
    current["state"] = state
    current["updated_at"] = entry["at"]
    current["history"] = history
    atomic_write_json(job.state_json, current)
    return current


def job_state(job: JobPaths) -> Optional[str]:
    current = read_json(job.state_json, default=None)
    return current.get("state") if isinstance(current, dict) else None


def free_bytes(path: "os.PathLike[str] | str") -> int:
    stats = os.statvfs(str(path))
    return stats.f_bavail * stats.f_frsize


# --------------------------------------------------------------------- #
# Logging, events and heartbeats
# --------------------------------------------------------------------- #
def setup_logging(service: str) -> logging.Logger:
    level = env_str("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level, logging.INFO),
        format=f"%(asctime)s %(levelname)s [{service}] %(message)s",
    )
    # pynetdicom is very chatty at INFO; keep association-level detail at DEBUG.
    logging.getLogger("pynetdicom").setLevel(
        logging.DEBUG if level == "DEBUG" else logging.WARNING
    )
    return logging.getLogger(service)


_EVENT_LOCK = threading.Lock()


def append_event(paths: DataPaths, service: str, event: str, **fields: Any) -> None:
    """Append one JSON line to ``audit/events.jsonl``.

    Callers must only pass technical identifiers (job id, UIDs, AE titles,
    counts, statuses).  Patient names and IDs are never logged.
    """
    record = {"at": utc_now_iso(), "service": service, "event": event}
    record.update({key: value for key, value in fields.items() if value is not None})
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    try:
        paths.audit.mkdir(parents=True, exist_ok=True)
        with _EVENT_LOCK:
            with open(paths.events_jsonl, "a", encoding="utf-8") as handle:
                handle.write(line)
    except OSError as exc:  # never let audit I/O take a service down
        LOG.warning("Could not append event %s: %s", event, exc)


def touch_heartbeat(paths: DataPaths, service: str) -> None:
    try:
        paths.heartbeats.mkdir(parents=True, exist_ok=True)
        target = paths.heartbeats / service
        tmp = target.with_suffix(".tmp")
        tmp.write_text(f"{time.time():.3f}\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        LOG.warning("Could not update heartbeat for %s: %s", service, exc)


def heartbeat_age_seconds(paths: DataPaths, service: str) -> Optional[float]:
    try:
        stamp = float((paths.heartbeats / service).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return max(0.0, time.time() - stamp)


# --------------------------------------------------------------------- #
# Queue / key-value broker
# --------------------------------------------------------------------- #
class RedisBroker:
    """Thin wrapper over redis-py implementing a reliable-queue pattern.

    ``reserve`` atomically moves an item to a processing list (BLMOVE); the
    item stays there until ``ack``.  After a crash, ``requeue_processing``
    moves orphaned items back to the head of the source queue.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def push(self, queue: str, payload: dict) -> None:
        self._client.rpush(queue, json.dumps(payload, sort_keys=True))

    def reserve(self, source: str, processing: str, timeout: float) -> "tuple[str, dict] | None":
        raw = self._client.blmove(source, processing, timeout, "LEFT", "RIGHT")
        if raw is None:
            return None
        return raw, json.loads(raw)

    def ack(self, processing: str, raw: str) -> None:
        self._client.lrem(processing, 1, raw)

    def requeue_processing(self, processing: str, source: str) -> int:
        moved = 0
        while self._client.lmove(processing, source, "RIGHT", "LEFT") is not None:
            moved += 1
        return moved

    def length(self, queue: str) -> int:
        return int(self._client.llen(queue))

    def items(self, queue: str) -> "list[dict]":
        return [json.loads(raw) for raw in self._client.lrange(queue, 0, -1)]

    def set_nx(self, key: str, value: str, ttl_seconds: int) -> bool:
        return bool(self._client.set(key, value, nx=True, ex=ttl_seconds))

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        self._client.set(key, value, ex=ttl_seconds)

    def set_keep_ttl(self, key: str, value: str) -> None:
        """Overwrite a value without resetting its expiry."""
        self._client.set(key, value, keepttl=True)

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def ping(self) -> bool:
        return bool(self._client.ping())


class InMemoryBroker:
    """Thread-safe stand-in for :class:`RedisBroker` used by unit tests."""

    def __init__(self) -> None:
        self._lists: "dict[str, list[str]]" = {}
        self._kv: "dict[str, str]" = {}
        self._cond = threading.Condition()

    def push(self, queue: str, payload: dict) -> None:
        with self._cond:
            self._lists.setdefault(queue, []).append(json.dumps(payload, sort_keys=True))
            self._cond.notify_all()

    def reserve(self, source: str, processing: str, timeout: float) -> "tuple[str, dict] | None":
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._lists.get(source):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            raw = self._lists[source].pop(0)
            self._lists.setdefault(processing, []).append(raw)
            return raw, json.loads(raw)

    def ack(self, processing: str, raw: str) -> None:
        with self._cond:
            items = self._lists.get(processing, [])
            if raw in items:
                items.remove(raw)

    def requeue_processing(self, processing: str, source: str) -> int:
        with self._cond:
            items = self._lists.pop(processing, [])
            self._lists[source] = items + self._lists.get(source, [])
            self._cond.notify_all()
            return len(items)

    def length(self, queue: str) -> int:
        with self._cond:
            return len(self._lists.get(queue, []))

    def items(self, queue: str) -> "list[dict]":
        with self._cond:
            return [json.loads(raw) for raw in self._lists.get(queue, [])]

    def set_nx(self, key: str, value: str, ttl_seconds: int) -> bool:
        with self._cond:
            if key in self._kv:
                return False
            self._kv[key] = value
            return True

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        with self._cond:
            self._kv[key] = value

    def set_keep_ttl(self, key: str, value: str) -> None:
        self.set(key, value)

    def get(self, key: str) -> Optional[str]:
        with self._cond:
            return self._kv.get(key)

    def delete(self, key: str) -> None:
        with self._cond:
            self._kv.pop(key, None)

    def ping(self) -> bool:
        return True


def connect_redis(logger: logging.Logger, *, wait_seconds: float = 120.0) -> RedisBroker:
    """Connect to Redis from REDIS_HOST/REDIS_PORT/REDIS_DB, retrying at startup."""
    import redis  # imported lazily so unit tests do not need redis-py

    host = env_str("REDIS_HOST", "prostate-redis")
    port = env_int("REDIS_PORT", 6379)
    db = env_int("REDIS_DB", 0)
    client = redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=30,          # must exceed the BLMOVE block timeout
        health_check_interval=30,
    )
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            client.ping()
            logger.info("Connected to Redis at %s:%s/%s", host, port, db)
            return RedisBroker(client)
        except Exception as exc:  # redis.exceptions.ConnectionError and friends
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Redis at {host}:{port} unreachable: {exc}") from exc
            logger.warning("Waiting for Redis at %s:%s (%s)", host, port, exc)
            time.sleep(2)
