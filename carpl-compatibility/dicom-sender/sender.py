import json
import logging
import os
import shutil

import pandas as pd
import pydicom
import redis
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pynetdicom import AE

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

# "local"      → C-STORE to the local carpl-receiver stub (data/test_carpl/)
# "production" → C-STORE to the real CARPL DICOM server
SENDER_MODE = os.environ.get("SENDER_MODE", "local")

if SENDER_MODE == "local":
    DEST_AE   = "CARPL_DICOM"
    DEST_HOST = "carpl-receiver"
    DEST_PORT = 11113
else:
    DEST_AE   = os.environ.get("DEST_AE",   "CARPL_DICOM")
    DEST_HOST = os.environ.get("DEST_HOST",  "10.10.30.50")
    DEST_PORT = int(os.environ.get("DEST_PORT", 11112))

MY_AE = os.environ.get("MY_AE", "AI_ENGINE")

PRIVATE_CREATOR = "CARPL_AI"
PRIVATE_GROUP = 0x0013

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("dicom-sender")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_timeout=None,
    socket_connect_timeout=10,
)

logger.info(f"Waiting for sender jobs... (mode: {SENDER_MODE.upper()} → {DEST_AE}@{DEST_HOST}:{DEST_PORT})")

while True:
    result = redis_client.blpop("dicom_sender_jobs", timeout=30)
    if result is None:
        continue

    _, raw = result
    job = json.loads(raw)

    job_uid = job["job_uid"]
    dicom_path = job["dicom_path"]
    results_csv = job["results_csv"]

    logger.info(f"[{job_uid}] Attaching findings to {dicom_path}")

    try:
        df = pd.read_csv(results_csv)
        if df.empty:
            raise ValueError("results.csv is empty — no inference output to send")

        row = df.iloc[0]
        probability = round(float(row["probability"]), 4)
        prediction = "positive" if int(row["prediction"]) == 1 else "negative"

        findings_payload = json.dumps({
            "findings": [
                {"name": "osteoporosis", "probability": probability}
            ]
        })

        ds = pydicom.dcmread(dicom_path)

        # Ensure file_meta exists with a valid transfer syntax
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()
            ds.file_meta = file_meta
            ds.is_implicit_VR = False
            ds.is_little_endian = True

        block = ds.private_block(PRIVATE_GROUP, PRIVATE_CREATOR, create=True)
        block.add_new(0x01, "LT", findings_payload)

        # Save the modified DICOM back to disk for audit
        ds.save_as(dicom_path, write_like_original=False)

        # Send via C-STORE SCU
        ae = AE(ae_title=MY_AE)
        ae.add_requested_context(ds.SOPClassUID)

        assoc = ae.associate(DEST_HOST, DEST_PORT, ae_title=DEST_AE)
        if not assoc.is_established:
            raise ConnectionError(
                f"Could not associate with {DEST_AE}@{DEST_HOST}:{DEST_PORT}"
            )

        status = assoc.send_c_store(ds)
        assoc.release()

        status_code = status.Status if status else None
        logger.info(f"[{job_uid}] C-STORE status: {hex(status_code) if status_code is not None else 'no response'}")

        if status_code == 0x0000:
            logger.info(
                f"[{job_uid}] C-STORE succeeded → {DEST_AE}@{DEST_HOST}:{DEST_PORT} "
                f"[{prediction}, p={probability}]"
            )
            output_dir = os.path.dirname(dicom_path)
            try:
                shutil.rmtree(output_dir)
                logger.info(f"[{job_uid}] Cleaned up {output_dir}")
            except Exception as cleanup_err:
                logger.error(f"[{job_uid}] Cleanup failed for {output_dir}: {cleanup_err}")
        else:
            logger.error(f"[{job_uid}] C-STORE failed — output retained at {os.path.dirname(dicom_path)}")

    except Exception as e:
        logger.error(f"[{job_uid}] Failed: {e}", exc_info=True)
