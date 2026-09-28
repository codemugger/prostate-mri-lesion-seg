import json
import os
import sys
import uuid
from datetime import datetime, timezone

import pydicom
import redis
from pynetdicom import AE, evt
from pynetdicom.sop_class import Verification

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/data/incoming")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")
DICOM_PORT = int(os.environ.get("DICOM_PORT", 11112))
DICOM_AE = os.environ.get("DICOM_AE", "CARPLSTORESCP")

# Created once at process startup — reused for every incoming DICOM
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
)


def handle_store(event):
    """Called once per received DICOM file."""
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta
    except Exception as e:
        print(f"[REJECTED] Could not parse DICOM dataset: {e}", file=sys.stderr)
        return 0xC000  # Failure status

    os.makedirs(STORAGE_DIR, exist_ok=True)
    filepath = os.path.join(STORAGE_DIR, f"{ds.SOPInstanceUID}.dcm")
    try:
        ds.save_as(filepath, write_like_original=False)
    except Exception as e:
        print(f"[ERROR] Could not save DICOM to {filepath}: {e}", file=sys.stderr)
        return 0xC000

    job_uid = str(uuid.uuid4())
    job = {
        "job_uid": job_uid,
        "study_uid": str(getattr(ds, "StudyInstanceUID", "unknown")),
        "input_path": filepath,
        "output_path": os.path.join(OUTPUT_DIR, job_uid),
        "calling_ae": event.assoc.requestor.ae_title.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    redis_client.rpush("dicom_jobs", json.dumps(job))
    print(f"[ACCEPTED] Queued job {job_uid}: {filepath}")
    return 0x0000  # Success


ae = AE(ae_title=DICOM_AE)
ae.add_supported_context(Verification)

# Accept all standard storage SOP classes
from pynetdicom.sop_class import _STORAGE_CLASSES
for uid in _STORAGE_CLASSES.values():
    ae.add_supported_context(uid)

handlers = [(evt.EVT_C_STORE, handle_store)]

print(f"Starting DICOM listener — AE: {DICOM_AE}, port: {DICOM_PORT}")
print(f"Redis: {REDIS_HOST}:{REDIS_PORT}  |  Storage: {STORAGE_DIR}")
ae.start_server(("", DICOM_PORT), evt_handlers=handlers, block=True)
