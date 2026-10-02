#!/usr/bin/env python3
"""Parity checks between the CARPL path and direct pipeline runs.

Run this inside the ai-worker image (it has nibabel, numpy, pydicom,
highdicom and PyYAML), with the repository mounted read-only::

    docker run --rm --entrypoint python \
        -v "$PWD":/repo:ro -v "$PWD/carpl/data":/data:ro -v /path/to/direct:/direct:ro \
        prostate-carpl-ai-worker:1.0.0 /repo/carpl/tools/parity_check.py \
        outputs /direct /data/jobs/<job_id>/output

Subcommands
-----------
outputs REFERENCE CURRENT
    Compare two single-case output folders:
    * every NIfTI: voxel arrays and affines exactly equal;
    * ``lesions.txt`` and ``organ/cleanup_metrics.json``: identical bytes;
    * ``case_status`` status equal;
    * the T2/ADC/HIGHB series actually selected;
    * the combined RTSTRUCT, SR and supporting RTSTRUCTs as DICOM, after
      replacing run-specific UIDs and dates/times with placeholders.
      ContourData and measurements must match exactly;
    * ``validate_rtstruct_outputs.validate_case`` on both, plus
      ``compare_batch_outputs.compare_case``.
transport SOURCE STORED
    For every DICOM file under SOURCE, find ``<SOPInstanceUID>.dcm`` under
    STORED and compare the encoded dataset bytes after the file meta.  This
    proves the listener and sender neither decode nor alter data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

SELECTED_ROLES = ("t2", "adc", "highb")
TEXT_FILES = ("lesions.txt", "organ/cleanup_metrics.json")
DICOM_FILES = (
    "combined_organ_lesion_RTSTRUCT.dcm",
    "prostate_measurements_SR.dcm",
    "organ/organ_RTSTRUCT.dcm",
    "organ/cleaned_organ_RTSTRUCT.dcm",
    "lesion/lesion_RTSTRUCT.dcm",
)


def _import_scripts(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import case_status  # noqa: F401
    return case_status


# --------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------- #
def compare_nifti(reference: Path, current: Path) -> "list[str]":
    import nibabel as nib
    import numpy as np

    problems = []
    rels = sorted(
        {p.relative_to(reference).as_posix() for p in reference.rglob("*.nii.gz")}
        | {p.relative_to(current).as_posix() for p in current.rglob("*.nii.gz")}
    )
    for rel in rels:
        a, b = reference / rel, current / rel
        if not (a.is_file() and b.is_file()):
            problems.append(f"{rel}: present only in {'reference' if a.is_file() else 'current'}")
            continue
        img_a, img_b = nib.load(str(a)), nib.load(str(b))
        arr_a, arr_b = np.asanyarray(img_a.dataobj), np.asanyarray(img_b.dataobj)
        if arr_a.shape != arr_b.shape:
            problems.append(f"{rel}: shape {arr_a.shape} != {arr_b.shape}")
        elif not np.array_equal(arr_a, arr_b):
            diff = int(np.count_nonzero(arr_a != arr_b))
            problems.append(f"{rel}: {diff} voxel(s) differ")
        if not np.array_equal(img_a.affine, img_b.affine):
            problems.append(f"{rel}: affine differs (max {float(np.max(np.abs(img_a.affine - img_b.affine))):.3g})")
    print(f"  NIfTI files compared: {len(rels)}")
    return problems


def compare_text(reference: Path, current: Path) -> "list[str]":
    problems = []
    for rel in TEXT_FILES:
        a, b = reference / rel, current / rel
        if not (a.is_file() and b.is_file()):
            if a.is_file() or b.is_file():
                problems.append(f"{rel}: present only in {'reference' if a.is_file() else 'current'}")
            continue
        if a.read_bytes() != b.read_bytes():
            problems.append(f"{rel}: contents differ")
    return problems


def selected_series(case_dir: Path) -> "dict[str, str | None]":
    import pydicom

    selected = {}
    for role in SELECTED_ROLES:
        files = sorted((case_dir / "dicom" / role).glob("*.dcm"))
        selected[role] = str(pydicom.dcmread(files[0], stop_before_pixels=True).SeriesInstanceUID) if files else None
    return selected


def _input_uids(case_dir: Path) -> "set[str]":
    import pydicom

    uids = set()
    for path in (case_dir / "dicom").rglob("*.dcm"):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        for keyword in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"):
            if keyword in ds:
                uids.add(str(ds[keyword].value))
    return uids


def normalised_json(path: Path, keep_uids: "set[str]") -> dict:
    """DICOM JSON with run-specific UIDs and dates/times replaced by placeholders."""
    import pydicom

    ds = pydicom.dcmread(path)

    def scrub(dataset) -> None:
        for element in dataset:
            if element.VR == "SQ":
                for item in element.value:
                    scrub(item)
            elif element.VR == "UI":
                values = element.value if isinstance(element.value, list) else [element.value]
                if any(str(v) not in keep_uids and not str(v).startswith("1.2.840.10008.") for v in values):
                    element.value = "<generated-uid>"
            elif element.VR in ("DA", "TM", "DT"):
                element.value = ""

    scrub(ds)
    return ds.to_json_dict()


def _first_difference(a, b, path: str = "") -> str:
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key}: present only in {'reference' if key in a else 'current'}"
            found = _first_difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
        return ""
    return "" if a == b else f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}"


def compare_dicom(reference: Path, current: Path) -> "list[str]":
    problems = []
    keep = _input_uids(reference) | _input_uids(current)
    for rel in DICOM_FILES:
        a, b = reference / rel, current / rel
        if not (a.is_file() and b.is_file()):
            if a.is_file() or b.is_file():
                problems.append(f"{rel}: present only in {'reference' if a.is_file() else 'current'}")
            continue
        difference = _first_difference(normalised_json(a, keep), normalised_json(b, keep))
        if difference:
            problems.append(f"{rel}: {difference}")
    return problems


def cmd_outputs(reference: Path, current: Path, scripts_dir: Path) -> int:
    case_status = _import_scripts(scripts_dir)
    problems: "list[str]" = []
    print(f"Reference: {reference}\nCurrent  : {current}")

    status_a = case_status.inspect_case(reference).status
    status_b = case_status.inspect_case(current).status
    print(f"  case_status: reference={status_a} current={status_b}")
    if status_a != status_b:
        problems.append(f"case_status differs: {status_a} != {status_b}")

    sel_a, sel_b = selected_series(reference), selected_series(current)
    for role in SELECTED_ROLES:
        same = sel_a[role] == sel_b[role]
        print(f"  selected {role:5s}: {'same' if same else 'DIFFERENT'} ({sel_a[role]} vs {sel_b[role]})")
        if not same:
            problems.append(f"different {role} series selected")

    problems += compare_nifti(reference, current)
    problems += compare_text(reference, current)
    problems += compare_dicom(reference, current)

    import validate_rtstruct_outputs
    import compare_batch_outputs

    for label, case in (("reference", reference), ("current", current)):
        row = validate_rtstruct_outputs.validate_case(case)
        print(f"  validate_rtstruct_outputs[{label}]: valid={row['valid']} "
              f"roundtrip_different_voxels={row.get('roundtrip_different_voxels')}")
        if row["valid"] != "true" and status_a.startswith("SUCCESS"):
            problems.append(f"validate_rtstruct_outputs failed for {label}: {row.get('error', row)}")
    comparison = compare_batch_outputs.compare_case(reference, current)
    print("  compare_batch_outputs: " + json.dumps(comparison, sort_keys=True)[:600])

    if problems:
        print("\nPARITY FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nPARITY OK: outputs are identical apart from run-specific UIDs and dates/times.")
    return 0


# --------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------- #
def dataset_bytes(path: Path) -> bytes:
    """Encoded dataset after the preamble and file meta group."""
    import pydicom
    from pydicom.filereader import _read_file_meta_info, read_preamble

    with open(path, "rb") as handle:
        read_preamble(handle, False)
        _read_file_meta_info(handle)
        return handle.read()


def cmd_transport(source: Path, stored: Path) -> int:
    import pydicom

    index = {p.stem: p for p in stored.rglob("*.dcm")}
    identical = different = missing = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        try:
            sop = str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
        except Exception:
            continue
        target = index.get(sop)
        if target is None:
            missing += 1
            continue
        if dataset_bytes(path) == dataset_bytes(target):
            identical += 1
        else:
            different += 1
            print(f"  DIFFERENT: {path} vs {target}")
    print(f"identical={identical} different={different} missing={missing}")
    return 0 if different == 0 and missing == 0 and identical > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    outputs = sub.add_parser("outputs")
    outputs.add_argument("reference", type=Path)
    outputs.add_argument("current", type=Path)
    outputs.add_argument("--scripts-dir", type=Path, default=REPO_SCRIPTS)
    transport = sub.add_parser("transport")
    transport.add_argument("source", type=Path)
    transport.add_argument("stored", type=Path)
    args = parser.parse_args()
    if args.command == "outputs":
        return cmd_outputs(args.reference, args.current, args.scripts_dir)
    return cmd_transport(args.source, args.stored)


if __name__ == "__main__":
    raise SystemExit(main())
