#!/usr/bin/env python3
"""Docker HEALTHCHECK for the CARPL services.

    healthcheck.py --service dicom-listener --max-age 60 --echo
    healthcheck.py --service ai-worker --max-age 120

Exits 0 when the service heartbeat is fresh and, with ``--echo``, when the
local Storage SCP answers a DICOM C-ECHO.  Exits 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.environ.get("CARPL_LIB_DIR", "/opt/carpl/lib"))

from carpl_common import data_paths_from_env, env_int, env_str, heartbeat_age_seconds  # noqa: E402

HEALTHCHECK_AE = "PROSTATE_HC"


def c_echo(port: int, called_ae: str) -> bool:
    from pynetdicom import AE
    from pynetdicom.sop_class import Verification

    ae = AE(ae_title=HEALTHCHECK_AE)
    ae.acse_timeout = 5
    ae.dimse_timeout = 5
    ae.network_timeout = 5
    ae.add_requested_context(Verification)
    assoc = ae.associate("127.0.0.1", port, ae_title=called_ae)
    if not assoc.is_established:
        return False
    try:
        status = assoc.send_c_echo()
        return bool(status) and status.Status == 0x0000
    finally:
        assoc.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True)
    parser.add_argument("--max-age", type=float, default=120.0)
    parser.add_argument("--echo", action="store_true")
    args = parser.parse_args()

    age = heartbeat_age_seconds(data_paths_from_env(), args.service)
    if age is None or age > args.max_age:
        print(f"unhealthy: heartbeat age {age}", file=sys.stderr)
        return 1
    if args.echo and not c_echo(env_int("DICOM_PORT", 11112), env_str("DICOM_AE", "PROSTATE_AI")):
        print("unhealthy: C-ECHO failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
