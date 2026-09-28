import os
import sys

import pydicom
from pynetdicom import AE, evt
from pynetdicom.sop_class import Verification, _STORAGE_CLASSES

STORAGE_DIR = os.environ.get("STORAGE_DIR", "/data/test_carpl")
DICOM_PORT = int(os.environ.get("DICOM_PORT", 11113))
DICOM_AE = os.environ.get("DICOM_AE", "CARPLLISTENER")


def handle_store(event):
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta
    except Exception as e:
        print(f"[REJECTED] Could not parse dataset: {e}", file=sys.stderr)
        return 0xC000

    os.makedirs(STORAGE_DIR, exist_ok=True)
    filepath = os.path.join(STORAGE_DIR, f"{ds.SOPInstanceUID}.dcm")
    try:
        ds.save_as(filepath, write_like_original=False)
        print(f"[STORED] {filepath}")
    except Exception as e:
        print(f"[ERROR] Could not save {filepath}: {e}", file=sys.stderr)
        return 0xC000

    return 0x0000


ae = AE(ae_title=DICOM_AE)
ae.add_supported_context(Verification)
for uid in _STORAGE_CLASSES.values():
    ae.add_supported_context(uid)

handlers = [(evt.EVT_C_STORE, handle_store)]

print(f"CARPL receiver — AE: {DICOM_AE}, port: {DICOM_PORT}, dir: {STORAGE_DIR}")
ae.start_server(("", DICOM_PORT), evt_handlers=handlers, block=True)
