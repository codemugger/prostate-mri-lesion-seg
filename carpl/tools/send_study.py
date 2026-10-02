#!/usr/bin/env python3
"""Test SCU: C-ECHO, then C-STORE a DICOM folder, simulating a CARPL push.

    python carpl/tools/send_study.py test-data/ProstateX-0004 \
        --host localhost --port 11112 --aec PROSTATE_AI

Each file is proposed with its own SOP class and transfer syntax and sent
from its path, so the bytes on the wire equal the source file's dataset.
``--include-series`` / ``--exclude-series`` (regex on SeriesDescription)
send part of a study, to test partial-study and late-arrival handling.

Requires pydicom and pynetdicom (``pip install -r carpl/requirements-carpl.txt``).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pydicom
from pynetdicom import AE
from pynetdicom.sop_class import Verification

MAX_CONTEXTS = 127  # 128 minus the Verification context


def collect(folder: Path, include: "re.Pattern | None", exclude: "re.Pattern | None"):
    items = []
    skipped = 0
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if path.name.upper() == "DICOMDIR":
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            sop_class = str(ds.SOPClassUID)
            transfer_syntax = str(ds.file_meta.TransferSyntaxUID)
        except Exception:
            skipped += 1
            continue
        description = str(ds.get("SeriesDescription", ""))
        if include and not include.search(description):
            continue
        if exclude and exclude.search(description):
            continue
        items.append((path, sop_class, transfer_syntax))
    return items, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=11112)
    parser.add_argument("--aec", default="PROSTATE_AI", help="called AE title (the listener)")
    parser.add_argument("--aet", default="CARPL_TEST", help="calling AE title (pretend CARPL)")
    parser.add_argument("--echo-only", action="store_true")
    parser.add_argument("--include-series", help="only send series whose SeriesDescription matches this regex")
    parser.add_argument("--exclude-series", help="skip series whose SeriesDescription matches this regex")
    args = parser.parse_args()

    ae = AE(ae_title=args.aet)
    ae.acse_timeout = ae.dimse_timeout = ae.network_timeout = 60
    ae.add_requested_context(Verification)

    items = []
    if not args.echo_only:
        include = re.compile(args.include_series, re.I) if args.include_series else None
        exclude = re.compile(args.exclude_series, re.I) if args.exclude_series else None
        items, skipped = collect(args.folder, include, exclude)
        contexts = sorted({(sop, ts) for _, sop, ts in items})
        if len(contexts) > MAX_CONTEXTS:
            print(f"Too many distinct SOP class/transfer syntax pairs ({len(contexts)})", file=sys.stderr)
            return 2
        for sop, ts in contexts:
            ae.add_requested_context(sop, [ts])
        print(f"Sending {len(items)} instance(s) from {args.folder} ({skipped} non-DICOM file(s) skipped)")

    assoc = ae.associate(args.host, args.port, ae_title=args.aec)
    if not assoc.is_established:
        print(f"Association with {args.aec}@{args.host}:{args.port} failed", file=sys.stderr)
        return 1
    try:
        echo = assoc.send_c_echo()
        echo_status = int(echo.Status) if echo and "Status" in echo else None
        print(f"C-ECHO status: {'0x%04X' % echo_status if echo_status is not None else 'no response'}")
        statuses: Counter = Counter()
        for path, _, _ in items:
            status = assoc.send_c_store(path)
            code = int(status.Status) if status and "Status" in status else None
            statuses["no response" if code is None else f"0x{code:04X}"] += 1
    finally:
        assoc.release()

    if items:
        print("C-STORE statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    ok = echo_status == 0 and all(key == "0x0000" for key in statuses)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
