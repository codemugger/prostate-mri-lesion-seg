#!/usr/bin/env python3
"""Local stand-in for CARPL's result receiver (development and testing only).

It mimics ``CARPL-RECIEVE@rapier-orthanc-prefect:4242``.  Every object is
stored exactly as received, so byte-level parity with the pipeline outputs
can be checked::

    /data/test_carpl/<StudyInstanceUID>/<SOPInstanceUID>.dcm

This container is never shipped in the production tar.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("CARPL_LIB_DIR", "/opt/carpl/lib"))

from carpl_common import (  # noqa: E402
    EXPLICIT_VR_LITTLE_ENDIAN,
    IMPLICIT_VR_LITTLE_ENDIAN,
    env_int,
    env_str,
    is_valid_uid,
    setup_logging,
    validate_ae_title,
)

SERVICE = "carpl-receiver"


def main() -> int:
    from pynetdicom import AE, AllStoragePresentationContexts, evt
    from pynetdicom.sop_class import Verification

    log: logging.Logger = setup_logging(SERVICE)
    ae_title = validate_ae_title(env_str("DICOM_AE", "CARPL-RECIEVE"), name="DICOM_AE")
    port = env_int("DICOM_PORT", 4242)
    storage = Path(env_str("STORAGE_DIR", "/data/test_carpl"))
    storage.mkdir(parents=True, exist_ok=True)

    def handle_store(event):
        ds = event.dataset
        study_uid = str(ds.get("StudyInstanceUID", ""))
        sop_uid = str(ds.get("SOPInstanceUID", ""))
        if not (is_valid_uid(study_uid) and is_valid_uid(sop_uid)):
            return 0xC000
        target = storage / study_uid / f"{sop_uid}.dcm"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(event.encoded_dataset())
        log.info(
            "Received %s (%s, series '%s') for study %s from %s",
            ds.SOPClassUID.name, ds.get("Modality", ""), ds.get("SeriesDescription", ""),
            study_uid, str(event.assoc.requestor.ae_title).strip(),
        )
        return 0x0000

    ae = AE(ae_title=ae_title)
    syntaxes = [EXPLICIT_VR_LITTLE_ENDIAN, IMPLICIT_VR_LITTLE_ENDIAN]
    ae.add_supported_context(Verification, syntaxes)
    for context in AllStoragePresentationContexts:
        ae.add_supported_context(context.abstract_syntax, syntaxes)
    log.info("CARPL receiver stub %s listening on port %d, storing to %s", ae_title, port, storage)
    ae.start_server(("", port), block=True, evt_handlers=[(evt.EVT_C_STORE, handle_store)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
