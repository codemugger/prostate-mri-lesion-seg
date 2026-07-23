"""Per-case status detection and CSV audit logging for the prostate MRI
lesion-segmentation batch pipeline.

Why this module exists
----------------------
When the pipeline runs over a large cohort (e.g., the ~3,500-case SGH
de-identified dataset on HIVE) several classes of "silent failures" are
possible:

  1. The MONAI ``DICOMSeriesSelectorOperator`` cannot find a required
     modality (T2 / ADC / HIGHB). The app still exits with code 0, but
     the per-case output directory ends up empty.
  2. The organ segmentation model returns an empty mask (no prostate
     detected); lesion inference is intentionally skipped and an
     all-zero lesion mask is written (see
     ``custom_lesion_seg_operator.py``). Exit code is 0 but no
     ``fold*_lesion_prob`` NIfTIs exist.
  3. A runtime crash inside lesion inference or classification.
  4. Pre-existing output in resume mode - we still want this case
     logged to keep the audit trail complete.

Without an explicit per-case audit trail, distinguishing "complete",
"empty but intentional" and "silently broken" requires forensic
inspection of hundreds of output folders after the fact. This module
fixes that by:

  * Inspecting the case's output directory for the expected
    NIfTI / RTSTRUCT / text files.
  * Classifying the outcome into a fixed vocabulary (see :class:`Status`).
  * Appending one row per case to a CSV audit log that is safe to
    consume by pandas / Excel and is resume-friendly.

The same inspection logic is used by both:

  * ``scripts/test_batch.sh``        (local non-Docker runner)
  * ``scripts/batch_load_and_run.py`` (Dockerised runner, used on HIVE)

so laptop and HIVE produce identical audit CSVs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------- #
# File layout expected by a successful case.
# Update here if the pipeline ever changes where it writes outputs.
# --------------------------------------------------------------------- #
EXPECTED_FILES: "dict[str, str]" = {
    "t2_nii":              "t2/t2.nii.gz",
    "adc_nii":             "adc/adc.nii.gz",
    "highb_nii":           "highb/highb.nii.gz",
    "organ_nii":           "organ/organ.nii.gz",
    "cleaned_organ_nii":   "organ/cleaned_organ.nii.gz",
    "cleanup_metrics":     "organ/cleanup_metrics.json",
    "organ_rtstruct":      "organ/organ_RTSTRUCT.dcm",
    "cleaned_organ_rtstruct": "organ/cleaned_organ_RTSTRUCT.dcm",
    "lesion_mask":         "lesion/lesion_mask.nii.gz",
    "lesion_rtstruct":     "lesion/lesion_RTSTRUCT.dcm",
    "merged_lesion_prob":  "lesion/merged_lesion_prob.nii.gz",
    "lesions_txt":         "lesions.txt",
    "prostate_sr":         "prostate_measurements_SR.dcm",
    "combined_rtstruct":   "combined_organ_lesion_RTSTRUCT.dcm",
}

# All five fold probability maps must exist for SUCCESS_COMPLETE.
LESION_FOLD_FILES = [f"lesion/fold{i}_lesion_prob.nii.gz" for i in range(5)]

# Columns in the audit CSV, in a stable order.
CSV_COLUMNS: "list[str]" = [
    "timestamp_utc",
    "patient_id",
    "status",
    "reason",
    "missing_modalities",
    "exit_code",
    "elapsed_seconds",
    "resume_skipped",
    "has_t2_nii",
    "has_adc_nii",
    "has_highb_nii",
    "has_organ_nii",
    "has_cleaned_organ_nii",
    "has_cleanup_metrics",
    "has_organ_rtstruct",
    "has_cleaned_organ_rtstruct",
    "has_lesion_mask",
    "has_lesion_rtstruct",
    "has_merged_lesion_prob",
    "has_lesion_folds",
    "has_lesions_txt",
    "has_prostate_sr",
    "has_combined_rtstruct",
    "organ_cleanup_status",
    "organ_island_count_before",
    "organ_removed_island_count",
    "organ_removed_voxels",
    "organ_filled_hole_voxels",
    "organ_largest_component_fraction",
    "output_file_count",
    "input_dir",
    "output_dir",
]


class Status:
    """Fixed vocabulary for the ``status`` column."""
    SUCCESS_COMPLETE      = "SUCCESS_COMPLETE"
    SUCCESS_EMPTY_LESION  = "SUCCESS_EMPTY_LESION"   # organ empty -> zero mask
    FAIL_MISSING_MODALITY = "FAIL_MISSING_MODALITY"  # selector rejected input
    FAIL_ORGAN_SEG        = "FAIL_ORGAN_SEG"         # no organ.nii.gz
    FAIL_POSTPROCESSING   = "FAIL_POSTPROCESSING"    # cleaned organ/metrics missing
    FAIL_LESION_SEG       = "FAIL_LESION_SEG"        # no lesion_mask.nii.gz
    FAIL_CLASSIFIER       = "FAIL_CLASSIFIER"        # no lesions.txt
    FAIL_REPORTING        = "FAIL_REPORTING"         # no DICOM SR
    FAIL_RTSTRUCT         = "FAIL_RTSTRUCT"          # one or more RTSTRUCTs missing
    FAIL_RUNTIME          = "FAIL_RUNTIME"           # non-zero exit code
    FAIL_EMPTY_OUTPUT     = "FAIL_EMPTY_OUTPUT"      # exit 0 but nothing at all
    SKIPPED_RESUME        = "SKIPPED_RESUME"         # already-present output


@dataclass
class CaseStatus:
    """Structured result of inspecting one case's output directory."""
    patient_id: str
    status: str
    reason: str
    missing_modalities: "list[str]" = field(default_factory=list)
    exit_code: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    resume_skipped: bool = False
    file_flags: "dict[str, bool]" = field(default_factory=dict)
    cleanup_metrics: "dict[str, object]" = field(default_factory=dict)
    output_file_count: int = 0
    input_dir: str = ""
    output_dir: str = ""
    timestamp_utc: str = ""

    def to_csv_row(self) -> "dict[str, str]":
        row: "dict[str, str]" = {
            "timestamp_utc": self.timestamp_utc,
            "patient_id": self.patient_id,
            "status": self.status,
            "reason": self.reason,
            "missing_modalities": ";".join(self.missing_modalities),
            "exit_code": "" if self.exit_code is None else str(self.exit_code),
            "elapsed_seconds": (
                "" if self.elapsed_seconds is None
                else f"{self.elapsed_seconds:.2f}"
            ),
            "resume_skipped": "true" if self.resume_skipped else "false",
            "organ_cleanup_status": str(
                self.cleanup_metrics.get("status", "")
            ),
            "organ_island_count_before": str(
                self.cleanup_metrics.get("original_component_count", "")
            ),
            "organ_removed_island_count": str(
                self.cleanup_metrics.get("removed_island_count", "")
            ),
            "organ_removed_voxels": str(
                self.cleanup_metrics.get("removed_voxels", "")
            ),
            "organ_filled_hole_voxels": str(
                self.cleanup_metrics.get("filled_hole_voxels", "")
            ),
            "organ_largest_component_fraction": str(
                self.cleanup_metrics.get("largest_component_fraction", "")
            ),
            "output_file_count": str(self.output_file_count),
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
        }
        for flag_name, present in self.file_flags.items():
            row[f"has_{flag_name}"] = "true" if present else "false"
        for col in CSV_COLUMNS:
            row.setdefault(col, "false" if col.startswith("has_") else "")
        return row


def _now_iso_utc() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# --------------------------------------------------------------------- #
# Core inspection
# --------------------------------------------------------------------- #
def inspect_case(
    case_output_dir: "os.PathLike[str] | str",
    *,
    patient_id: Optional[str] = None,
    input_dir: "os.PathLike[str] | str" = "",
    exit_code: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    resume_skipped: bool = False,
) -> CaseStatus:
    """Inspect a case's output directory and return a :class:`CaseStatus`.

    Status is decided by file presence, *not* by exit code (since the app
    regularly exits 0 even when the pipeline produced nothing). The exit
    code only overrides to ``FAIL_RUNTIME`` when non-zero and we did not
    skip the case via resume.
    """
    case_output_dir = Path(case_output_dir)
    pid = patient_id or case_output_dir.name

    flags: "dict[str, bool]" = {
        key: (case_output_dir / rel).is_file()
        for key, rel in EXPECTED_FILES.items()
    }
    flags["lesion_folds"] = all(
        (case_output_dir / rel).is_file() for rel in LESION_FOLD_FILES
    )

    cleanup_metrics: "dict[str, object]" = {}
    metrics_path = case_output_dir / EXPECTED_FILES["cleanup_metrics"]
    if metrics_path.is_file():
        try:
            with open(metrics_path, encoding="utf-8") as metrics_file:
                cleanup_metrics = json.load(metrics_file)
            if int(cleanup_metrics.get("original_foreground_voxels", 0)) == 0:
                cleanup_metrics["status"] = "EMPTY"
            elif (
                int(cleanup_metrics.get("removed_voxels", 0)) > 0
                or int(cleanup_metrics.get("filled_hole_voxels", 0)) > 0
            ):
                cleanup_metrics["status"] = "CLEANED"
            else:
                cleanup_metrics["status"] = "UNCHANGED"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            flags["cleanup_metrics"] = False
            cleanup_metrics = {"status": "INVALID"}

    file_count = 0
    if case_output_dir.exists():
        for p in case_output_dir.rglob("*"):
            if p.is_file():
                file_count += 1

    missing_mods: "list[str]" = []
    for mod, key in (("t2", "t2_nii"), ("adc", "adc_nii"), ("highb", "highb_nii")):
        if not flags[key]:
            missing_mods.append(mod)

    if exit_code is not None and exit_code != 0 and not resume_skipped:
        status = Status.FAIL_RUNTIME
        reason = (
            f"App exited with non-zero code {exit_code}. "
            + _describe_file_state(flags, missing_mods, file_count)
        )
    else:
        status, reason = _classify_from_files(flags, missing_mods, file_count)
        if resume_skipped:
            # Promote to SKIPPED_RESUME only when the existing output is a
            # genuine success; otherwise keep the diagnostic FAIL_* status
            # so an incomplete previous run is still flagged on resume.
            if status in (Status.SUCCESS_COMPLETE, Status.SUCCESS_EMPTY_LESION):
                status = Status.SKIPPED_RESUME
                reason = f"[resume] Existing output re-used. {reason}"
            else:
                reason = f"[resume] {reason}"

    return CaseStatus(
        patient_id=pid,
        status=status,
        reason=reason,
        missing_modalities=missing_mods,
        exit_code=exit_code,
        elapsed_seconds=elapsed_seconds,
        resume_skipped=resume_skipped,
        file_flags=flags,
        cleanup_metrics=cleanup_metrics,
        output_file_count=file_count,
        input_dir=str(input_dir) if input_dir else "",
        output_dir=str(case_output_dir),
        timestamp_utc=_now_iso_utc(),
    )


def _describe_file_state(
    flags: "dict[str, bool]",
    missing_mods: "list[str]",
    file_count: int,
) -> str:
    if file_count == 0:
        return "No output files were produced."
    if missing_mods:
        return f"Missing modality NIfTIs: {', '.join(missing_mods)}."
    if not flags["organ_nii"]:
        return "Modality NIfTIs present but organ segmentation missing."
    if not flags["cleaned_organ_nii"] or not flags["cleanup_metrics"]:
        return "Original organ mask present but cleaned organ output missing."
    if not flags["lesion_mask"]:
        return "Organ segmentation present but lesion segmentation missing."
    if not flags["lesions_txt"]:
        return "Lesion mask present but classifier report (lesions.txt) missing."
    if not flags["prostate_sr"]:
        return "Text report present but prostate DICOM SR missing."
    return "Partial output present."


def _classify_from_files(
    flags: "dict[str, bool]",
    missing_mods: "list[str]",
    file_count: int,
) -> "tuple[str, str]":
    """Classify outcome based solely on file presence."""
    if file_count == 0:
        return (
            Status.FAIL_EMPTY_OUTPUT,
            "No output files produced at all - most likely every series "
            "selector rejected the input (check SeriesDescription regexes "
            "in prostate_mri_lesion_seg_app/app.py).",
        )

    if missing_mods:
        return (
            Status.FAIL_MISSING_MODALITY,
            f"Missing intermediate NIfTI for: {', '.join(missing_mods)}. "
            f"DICOMSeriesSelectorOperator produced no match for these "
            f"modalities.",
        )

    if not flags["organ_nii"]:
        return (
            Status.FAIL_ORGAN_SEG,
            "All modality NIfTIs present but organ.nii.gz is missing - "
            "the organ segmentation operator did not produce output.",
        )

    if not flags["cleaned_organ_nii"] or not flags["cleanup_metrics"]:
        return (
            Status.FAIL_POSTPROCESSING,
            "Original organ mask exists but cleaned_organ.nii.gz or "
            "cleanup_metrics.json is missing.",
        )

    if not flags["lesion_mask"]:
        return (
            Status.FAIL_LESION_SEG,
            "Organ segmentation ok but lesion_mask.nii.gz is missing.",
        )

    if not flags["lesions_txt"]:
        return (
            Status.FAIL_CLASSIFIER,
            "Lesion mask produced but lesions.txt (classifier report) "
            "is missing.",
        )

    if not flags["prostate_sr"]:
        return (
            Status.FAIL_REPORTING,
            "lesions.txt exists but prostate_measurements_SR.dcm is missing.",
        )

    rtstruct_flags = (
        "organ_rtstruct",
        "cleaned_organ_rtstruct",
        "lesion_rtstruct",
        "combined_rtstruct",
    )
    missing_rtstructs = [name for name in rtstruct_flags if not flags[name]]
    if missing_rtstructs:
        return (
            Status.FAIL_RTSTRUCT,
            "Missing expected RTSTRUCT output(s): "
            + ", ".join(missing_rtstructs),
        )

    if flags["merged_lesion_prob"] and flags["lesion_folds"]:
        return (
            Status.SUCCESS_COMPLETE,
            "All modalities selected; organ cleanup, lesion inference, "
            "RTSTRUCT export, text report, and DICOM SR completed.",
        )

    return (
        Status.SUCCESS_EMPTY_LESION,
        "Organ mask was empty (no prostate detected); lesion inference "
        "was intentionally skipped and an empty lesion mask was written "
        "(see custom_lesion_seg_operator.py).",
    )


# --------------------------------------------------------------------- #
# CSV audit log (append-safe, header-safe)
# --------------------------------------------------------------------- #
def append_csv_row(csv_path: "os.PathLike[str] | str", case: CaseStatus) -> None:
    """Append one row, migrating an older audit schema when necessary."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_columns = reader.fieldnames or []
            if existing_columns != CSV_COLUMNS:
                existing_rows = list(reader)
                temporary_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
                with open(
                    temporary_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as migrated_file:
                    writer = csv.DictWriter(
                        migrated_file,
                        fieldnames=CSV_COLUMNS,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow({
                            column: row.get(column, "")
                            for column in CSV_COLUMNS
                        })
                temporary_path.replace(csv_path)

    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(case.to_csv_row())


def already_logged(csv_path: "os.PathLike[str] | str", patient_id: str) -> bool:
    """Return True if a row for ``patient_id`` already exists in the CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("patient_id") == patient_id:
                    return True
    except (OSError, csv.Error):
        return False
    return False


def summarise(csv_path: "os.PathLike[str] | str") -> "dict[str, int]":
    """Return a ``{status: count}`` summary of the audit CSV."""
    csv_path = Path(csv_path)
    counts: "dict[str, int]" = {}
    if not csv_path.exists():
        return counts
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "") or "UNKNOWN"
            counts[status] = counts.get(status, 0) + 1
    return counts


# --------------------------------------------------------------------- #
# CLI entrypoint (used by scripts/test_batch.sh)
# --------------------------------------------------------------------- #
def _cli(argv: "Optional[list[str]]" = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a case's output directory and append a row to the "
            "batch audit CSV. Used by scripts/test_batch.sh; the same "
            "classification function is imported directly by "
            "scripts/batch_load_and_run.py."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    log_parser = sub.add_parser(
        "log", help="Inspect one case and append a CSV row (default)."
    )
    for p in (parser, log_parser):
        p.add_argument("--case-output-dir", type=Path)
        p.add_argument("--patient-id", type=str)
        p.add_argument("--csv", type=Path)
        p.add_argument("--input-dir", type=Path, default=Path(""))
        p.add_argument("--exit-code", type=int, default=None)
        p.add_argument("--elapsed-seconds", type=float, default=None)
        p.add_argument("--resume-skipped", action="store_true")
        p.add_argument("--skip-if-logged", action="store_true",
                       help="No-op if a row for this patient_id already "
                            "exists in the CSV (resume-safe).")
        p.add_argument("--json-out", action="store_true",
                       help="Emit a compact JSON summary on stdout in "
                            "addition to the human-readable line.")

    summary_parser = sub.add_parser(
        "summary", help="Print a status breakdown of an existing audit CSV."
    )
    summary_parser.add_argument("csv_path", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "summary":
        counts = summarise(args.csv_path)
        total = sum(counts.values())
        print(f"Audit CSV: {args.csv_path}")
        print(f"Total rows: {total}")
        for status in sorted(counts):
            print(f"  {status:<24} {counts[status]:>5}")
        return 0

    # Default path: log one case.
    if not (args.case_output_dir and args.patient_id and args.csv):
        parser.error(
            "--case-output-dir, --patient-id and --csv are required "
            "unless running the 'summary' subcommand."
        )

    if args.skip_if_logged and already_logged(args.csv, args.patient_id):
        if args.json_out:
            print(json.dumps({
                "patient_id": args.patient_id,
                "status": "ALREADY_LOGGED",
            }))
        else:
            print(f"  Audit: row for '{args.patient_id}' already logged; "
                  f"not re-appending.")
        return 0

    case = inspect_case(
        args.case_output_dir,
        patient_id=args.patient_id,
        input_dir=args.input_dir,
        exit_code=args.exit_code,
        elapsed_seconds=args.elapsed_seconds,
        resume_skipped=args.resume_skipped,
    )
    append_csv_row(args.csv, case)

    if args.json_out:
        print(json.dumps({
            "patient_id": case.patient_id,
            "status": case.status,
            "reason": case.reason,
            "missing_modalities": case.missing_modalities,
            "exit_code": case.exit_code,
            "elapsed_seconds": case.elapsed_seconds,
        }))
    else:
        miss = (
            f"  [missing: {','.join(case.missing_modalities)}]"
            if case.missing_modalities else ""
        )
        print(f"  Status: {case.status}{miss}")
        print(f"  Reason: {case.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
