"""Compare a new batch against a prior pipeline output without ground truth.

This is a regression/audit comparison, not an accuracy evaluation. It verifies
that input conversion and the retained original organ mask did not change, and
quantifies expected downstream changes caused by cleaned-organ post-processing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import highdicom as hd
import nibabel as nib
import numpy as np
import pydicom
import yaml
from pydicom.uid import (
    EnhancedMRImageStorage,
    LegacyConvertedEnhancedMRImageStorage,
    MRImageStorage,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from case_status import inspect_case  # noqa: E402


CSV_COLUMNS = [
    "patient_id",
    "current_status",
    "t2_exact",
    "adc_exact",
    "highb_exact",
    "original_organ_exact",
    "original_organ_dice",
    "cleaned_vs_original_organ_dice",
    "organ_removed_voxels",
    "organ_filled_hole_voxels",
    "lesion_mask_dice",
    "old_lesion_voxels",
    "new_lesion_voxels",
    "old_lesion_count",
    "new_lesion_count",
    "old_pi_rads",
    "new_pi_rads",
    "merged_probability_mae",
    "dicom_sr_valid",
    "dicom_sr_study_matches_t2",
    "dicom_sr_measurements_match_yaml",
    "dicom_sr_full_t2_evidence",
    "dicom_sr_evidence_image_count",
    "t2_image_count",
    "notes",
]

MR_IMAGE_STORAGE_CLASSES = {
    str(MRImageStorage),
    str(EnhancedMRImageStorage),
    str(LegacyConvertedEnhancedMRImageStorage),
}


def _load_array(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.asarray(nib.load(str(path)).get_fdata())


def _exact(reference: Path, current: Path) -> str:
    old = _load_array(reference)
    new = _load_array(current)
    if old is None or new is None:
        return ""
    return str(old.shape == new.shape and np.array_equal(old, new)).lower()


def _dice(old: np.ndarray | None, new: np.ndarray | None) -> str:
    if old is None or new is None or old.shape != new.shape:
        return ""
    old_binary = old > 0
    new_binary = new > 0
    denominator = int(old_binary.sum() + new_binary.sum())
    value = (
        1.0
        if denominator == 0
        else 2.0 * int(np.logical_and(old_binary, new_binary).sum())
        / denominator
    )
    return f"{value:.6f}"


def _lesion_report(path: Path) -> tuple[int, str]:
    if not path.is_file():
        return 0, ""
    with open(path, encoding="utf-8") as report_file:
        content = yaml.safe_load(report_file) or []
    lesions = [
        item for item in content
        if isinstance(item, dict) and "Lesion_ID" in item
    ]
    grades = sorted(
        str(item["PI_RADS"]) for item in lesions if "PI_RADS" in item
    )
    return len(lesions), ";".join(grades)


def _organ_report(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as report_file:
        content = yaml.safe_load(report_file) or []
    for item in content:
        if isinstance(item, dict) and item.get("Organ") == "Prostate":
            return item
    return {}


def _validate_sr(case_directory: Path) -> dict[str, str]:
    sr_path = case_directory / "prostate_measurements_SR.dcm"
    if not sr_path.is_file():
        return {
            "valid": "false",
            "study_matches": "false",
            "measurements_match": "false",
            "full_evidence": "false",
            "evidence_count": "0",
            "t2_image_count": "0",
        }
    try:
        sr = pydicom.dcmread(str(sr_path), stop_before_pixels=True)
        hd.sr.ComprehensiveSR.from_dataset(sr)
        valid = (
            str(sr.Modality) == "SR"
            and str(sr.CompletionFlag) == "COMPLETE"
            and str(sr.VerificationFlag) == "UNVERIFIED"
        )
        t2_files = sorted((case_directory / "dicom" / "t2").glob("*.dcm"))
        if not t2_files:
            raise ValueError("No copied T2 evidence")
        t2_images = []
        for path in t2_files:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )
            if (
                str(getattr(dataset, "Modality", "")) == "MR"
                and str(getattr(dataset, "SOPClassUID", ""))
                in MR_IMAGE_STORAGE_CLASSES
            ):
                t2_images.append(dataset)
        if not t2_images:
            raise ValueError("No MR image storage objects in copied T2 evidence")
        t2 = t2_images[0]
        study_matches = (
            str(sr.StudyInstanceUID) == str(t2.StudyInstanceUID)
        )

        evidence_sops: set[str] = set()
        for sequence_name in (
            "CurrentRequestedProcedureEvidenceSequence",
            "PertinentOtherEvidenceSequence",
        ):
            for study_item in getattr(sr, sequence_name, []):
                for series_item in getattr(
                    study_item,
                    "ReferencedSeriesSequence",
                    [],
                ):
                    for sop_item in getattr(
                        series_item,
                        "ReferencedSOPSequence",
                        [],
                    ):
                        evidence_sops.add(
                            str(sop_item.ReferencedSOPInstanceUID)
                        )
        expected_sops = {
            str(dataset.SOPInstanceUID) for dataset in t2_images
        }
        full_evidence = evidence_sops == expected_sops

        sr_values: dict[tuple[str, str], float] = {}
        def collect_measurements(dataset, data_element):
            if (
                data_element.keyword == "MeasuredValueSequence"
                and hasattr(dataset, "ConceptNameCodeSequence")
            ):
                concept = dataset.ConceptNameCodeSequence[0]
                measured = data_element.value[0]
                unit = measured.MeasurementUnitsCodeSequence[0]
                sr_values[(str(concept.CodeValue), str(unit.CodeValue))] = (
                    float(measured.NumericValue)
                )
        sr.walk(collect_measurements)

        report = _organ_report(case_directory / "lesions.txt")
        dimensions = report.get("Dimensions_mm", {})
        expected_values = {
            ("PMLS001", "mm"): float(dimensions["Left_Right"]),
            ("PMLS002", "mm"): float(dimensions["Anterior_Posterior"]),
            ("PMLS003", "mm"): float(dimensions["Superior_Inferior"]),
            ("118565006", "mm3"): float(report["Volume_mm3"]),
            ("118565006", "cm3"): float(report["Volume_cc"]),
        }
        measurements_match = all(
            key in sr_values
            and abs(sr_values[key] - expected_value) <= 0.01
            for key, expected_value in expected_values.items()
        )
        return {
            "valid": str(valid).lower(),
            "study_matches": str(study_matches).lower(),
            "measurements_match": str(measurements_match).lower(),
            "full_evidence": str(full_evidence).lower(),
            "evidence_count": str(len(evidence_sops)),
            "t2_image_count": str(len(expected_sops)),
        }
    except Exception:
        return {
            "valid": "false",
            "study_matches": "false",
            "measurements_match": "false",
            "full_evidence": "false",
            "evidence_count": "0",
            "t2_image_count": "0",
        }


def compare_case(reference: Path, current: Path) -> dict[str, str]:
    patient_id = current.name
    notes: list[str] = []
    status = inspect_case(current, patient_id=patient_id)

    old_organ = _load_array(reference / "organ" / "organ.nii.gz")
    new_original_organ = _load_array(current / "organ" / "organ.nii.gz")
    cleaned_organ = _load_array(current / "organ" / "cleaned_organ.nii.gz")
    old_lesion = _load_array(reference / "lesion" / "lesion_mask.nii.gz")
    new_lesion = _load_array(current / "lesion" / "lesion_mask.nii.gz")

    metrics = status.cleanup_metrics
    old_count, old_grades = _lesion_report(reference / "lesions.txt")
    new_count, new_grades = _lesion_report(current / "lesions.txt")
    sr_validation = _validate_sr(current)

    old_probability = _load_array(
        reference / "lesion" / "merged_lesion_prob.nii.gz"
    )
    new_probability = _load_array(
        current / "lesion" / "merged_lesion_prob.nii.gz"
    )
    probability_mae = ""
    if (
        old_probability is not None
        and new_probability is not None
        and old_probability.shape == new_probability.shape
    ):
        probability_mae = (
            f"{float(np.mean(np.abs(old_probability - new_probability))):.8f}"
        )

    if old_organ is None:
        notes.append("no_reference_organ")
    if new_original_organ is None:
        notes.append("no_current_original_organ")
    if old_lesion is None:
        notes.append("no_reference_lesion")
    if new_lesion is None:
        notes.append("no_current_lesion")

    return {
        "patient_id": patient_id,
        "current_status": status.status,
        "t2_exact": _exact(
            reference / "t2" / "t2.nii.gz",
            current / "t2" / "t2.nii.gz",
        ),
        "adc_exact": _exact(
            reference / "adc" / "adc.nii.gz",
            current / "adc" / "adc.nii.gz",
        ),
        "highb_exact": _exact(
            reference / "highb" / "highb.nii.gz",
            current / "highb" / "highb.nii.gz",
        ),
        "original_organ_exact": _exact(
            reference / "organ" / "organ.nii.gz",
            current / "organ" / "organ.nii.gz",
        ),
        "original_organ_dice": _dice(old_organ, new_original_organ),
        "cleaned_vs_original_organ_dice": _dice(
            new_original_organ,
            cleaned_organ,
        ),
        "organ_removed_voxels": str(metrics.get("removed_voxels", "")),
        "organ_filled_hole_voxels": str(
            metrics.get("filled_hole_voxels", "")
        ),
        "lesion_mask_dice": _dice(old_lesion, new_lesion),
        "old_lesion_voxels": (
            "" if old_lesion is None else str(int((old_lesion > 0).sum()))
        ),
        "new_lesion_voxels": (
            "" if new_lesion is None else str(int((new_lesion > 0).sum()))
        ),
        "old_lesion_count": str(old_count),
        "new_lesion_count": str(new_count),
        "old_pi_rads": old_grades,
        "new_pi_rads": new_grades,
        "merged_probability_mae": probability_mae,
        "dicom_sr_valid": sr_validation["valid"],
        "dicom_sr_study_matches_t2": sr_validation["study_matches"],
        "dicom_sr_measurements_match_yaml": (
            sr_validation["measurements_match"]
        ),
        "dicom_sr_full_t2_evidence": sr_validation["full_evidence"],
        "dicom_sr_evidence_image_count": sr_validation["evidence_count"],
        "t2_image_count": sr_validation["t2_image_count"],
        "notes": ";".join(notes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Default: <current>/quality_comparison.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Default: <current>/quality_summary.json",
    )
    args = parser.parse_args()

    csv_path = args.csv or args.current / "quality_comparison.csv"
    summary_path = (
        args.summary_json or args.current / "quality_summary.json"
    )
    case_directories = sorted(
        path
        for path in args.current.iterdir()
        if path.is_dir() and (path / "lesions.txt").is_file()
    )
    rows = [
        compare_case(args.reference / case.name, case)
        for case in case_directories
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(row["current_status"] for row in rows)
    comparable_rows = [
        row for row in rows if row["original_organ_dice"] != ""
    ]
    summary = {
        "reference": str(args.reference),
        "current": str(args.current),
        "case_count": len(rows),
        "comparable_case_count": len(comparable_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "input_nifti_regressions": [
            row["patient_id"]
            for row in comparable_rows
            if any(
                row[field] == "false"
                for field in (
                    "t2_exact",
                    "adc_exact",
                    "highb_exact",
                )
            )
        ],
        "original_organ_nonidentical": [
            row["patient_id"]
            for row in comparable_rows
            if row["original_organ_exact"] == "false"
        ],
        "original_organ_material_changes": [
            row["patient_id"]
            for row in comparable_rows
            if row["original_organ_dice"] != ""
            and float(row["original_organ_dice"]) < 0.999
        ],
        "input_or_original_organ_regressions": [
            row["patient_id"]
            for row in comparable_rows
            if (
                any(
                    row[field] == "false"
                    for field in ("t2_exact", "adc_exact", "highb_exact")
                )
                or (
                    row["original_organ_dice"] != ""
                    and float(row["original_organ_dice"]) < 0.999
                )
            )
        ],
        "lesion_masks_changed": [
            row["patient_id"]
            for row in comparable_rows
            if row["lesion_mask_dice"] not in ("", "1.000000")
        ],
        "invalid_or_mismatched_dicom_sr": [
            row["patient_id"]
            for row in rows
            if row["current_status"] in (
                "SUCCESS_COMPLETE",
                "SUCCESS_EMPTY_LESION",
            )
            and (
                row["dicom_sr_valid"] != "true"
                or row["dicom_sr_study_matches_t2"] != "true"
                or row["dicom_sr_measurements_match_yaml"] != "true"
                or row["dicom_sr_full_t2_evidence"] != "true"
            )
        ],
        "interpretation": (
            "This is a regression comparison without ground truth. A changed "
            "lesion mask is not proof of improvement or deterioration and "
            "requires radiologist review."
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
