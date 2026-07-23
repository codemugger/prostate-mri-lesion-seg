"""Validate generated RTSTRUCT files against their source masks and T2 images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.uid import (
    EnhancedMRImageStorage,
    LegacyConvertedEnhancedMRImageStorage,
    MRImageStorage,
    RTStructureSetStorage,
)
from rt_utils import RTStructBuilder


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "prostate_mri_lesion_seg_app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rtstruct_utils import (  # noqa: E402
    _contour_area_mm2,
    _ensure_study_id,
    _match_mask_to_dicom_slices,
    _prepare_mask_from_array,
)


CSV_COLUMNS = [
    "patient_id",
    "valid",
    "rtstruct_count",
    "contour_count",
    "all_closedplanar_xor",
    "minimum_contour_points",
    "degenerate_contour_count",
    "roundtrip_exact",
    "roundtrip_different_voxels",
    "study_matches_t2",
    "all_t2_images_referenced",
    "contour_images_reference_t2",
    "invalid_short_string_count",
    "errors",
]

MR_IMAGE_STORAGE_CLASSES = {
    str(MRImageStorage),
    str(EnhancedMRImageStorage),
    str(LegacyConvertedEnhancedMRImageStorage),
}


def _prepared_mask(data: np.ndarray, slices: int) -> np.ndarray:
    return _match_mask_to_dicom_slices(
        _prepare_mask_from_array(data),
        slices,
    )


def _organ_rois(path: Path, slices: int) -> dict[str, np.ndarray]:
    data = np.asarray(nib.load(str(path)).get_fdata())
    if int(data.max()) >= 2:
        candidates = {
            "Prostate_TZ": _prepared_mask(data == 1, slices),
            "Prostate_PZ": _prepared_mask(data == 2, slices),
        }
    else:
        candidates = {"Prostate": _prepared_mask(data > 0, slices)}
    return {name: mask for name, mask in candidates.items() if mask.any()}


def _active_roi_names(dataset) -> set[str]:
    names_by_number = {
        int(item.ROINumber): str(item.ROIName)
        for item in getattr(dataset, "StructureSetROISequence", [])
    }
    return {
        names_by_number[int(item.ReferencedROINumber)]
        for item in getattr(dataset, "ROIContourSequence", [])
        if getattr(item, "ContourSequence", [])
    }


def _referenced_t2_sops(dataset) -> set[str]:
    referenced: set[str] = set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                for image in getattr(series, "ContourImageSequence", []):
                    referenced.add(str(image.ReferencedSOPInstanceUID))
    return referenced


def _invalid_short_strings(dataset) -> int:
    invalid = 0

    def collect(_parent, data_element):
        nonlocal invalid
        if data_element.VR != "SH":
            return
        values = (
            data_element.value
            if isinstance(data_element.value, (list, tuple))
            else [data_element.value]
        )
        invalid += sum(len(str(value)) > 16 for value in values)

    dataset.walk(collect)
    return invalid


def validate_case(case_directory: Path) -> dict[str, str]:
    errors: list[str] = []
    t2_directory = case_directory / "dicom" / "t2"
    t2_files = sorted(t2_directory.glob("*.dcm"))
    t2_datasets = [
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        for path in t2_files
    ]
    if not t2_datasets:
        raise ValueError(f"No T2 DICOM images found in {t2_directory}")
    t2_images = [
        dataset
        for dataset in t2_datasets
        if (
            str(getattr(dataset, "Modality", "")) == "MR"
            and str(getattr(dataset, "SOPClassUID", ""))
            in MR_IMAGE_STORAGE_CLASSES
        )
    ]
    if not t2_images:
        raise ValueError(f"No MR image objects found in {t2_directory}")
    t2_sops = {str(dataset.SOPInstanceUID) for dataset in t2_images}
    t2_study_uid = str(t2_images[0].StudyInstanceUID)

    patched_path, temporary_directory = _ensure_study_id(str(t2_directory))
    try:
        source = RTStructBuilder.create_new(dicom_series_path=patched_path)
        slice_count = len(source.series_data)
        lesion_data = np.asarray(
            nib.load(
                str(case_directory / "lesion" / "lesion_mask.nii.gz")
            ).get_fdata()
        )
        lesion_mask = _prepared_mask(lesion_data > 0, slice_count)
        original_organ = _organ_rois(
            case_directory / "organ" / "organ.nii.gz",
            slice_count,
        )
        cleaned_organ = _organ_rois(
            case_directory / "organ" / "cleaned_organ.nii.gz",
            slice_count,
        )

        rtstruct_specs = [
            (
                case_directory / "organ" / "organ_RTSTRUCT.dcm",
                original_organ,
            ),
            (
                case_directory
                / "organ"
                / "cleaned_organ_RTSTRUCT.dcm",
                cleaned_organ,
            ),
            (
                case_directory / "lesion" / "lesion_RTSTRUCT.dcm",
                {"Lesion": lesion_mask},
            ),
            (
                case_directory / "combined_organ_lesion_RTSTRUCT.dcm",
                {
                    **cleaned_organ,
                    **({"Lesion": lesion_mask} if lesion_mask.any() else {}),
                },
            ),
        ]

        contour_count = 0
        minimum_points: int | None = None
        degenerate_count = 0
        different_voxels = 0
        all_xor = True
        studies_match = True
        full_t2_reference = True
        contour_references_t2 = True
        invalid_short_strings = 0

        for rtstruct_path, expected_rois in rtstruct_specs:
            if not rtstruct_path.is_file():
                errors.append(f"missing:{rtstruct_path.name}")
                continue
            dataset = pydicom.dcmread(
                rtstruct_path,
                stop_before_pixels=True,
            )
            if str(dataset.SOPClassUID) != str(RTStructureSetStorage):
                errors.append(f"wrong_sop_class:{rtstruct_path.name}")
            studies_match &= str(dataset.StudyInstanceUID) == t2_study_uid
            full_t2_reference &= _referenced_t2_sops(dataset) == t2_sops
            invalid_short_strings += _invalid_short_strings(dataset)

            expected_active = {
                name for name, mask in expected_rois.items() if mask.any()
            }
            if _active_roi_names(dataset) != expected_active:
                errors.append(f"roi_mismatch:{rtstruct_path.name}")

            for roi_contour in getattr(dataset, "ROIContourSequence", []):
                for contour in getattr(
                    roi_contour,
                    "ContourSequence",
                    [],
                ):
                    contour_count += 1
                    all_xor &= (
                        str(contour.ContourGeometricType)
                        == "CLOSEDPLANAR_XOR"
                    )
                    points = np.asarray(
                        contour.ContourData,
                        dtype=np.float64,
                    ).reshape(-1, 3)
                    point_count = len(points)
                    minimum_points = (
                        point_count
                        if minimum_points is None
                        else min(minimum_points, point_count)
                    )
                    is_degenerate = (
                        point_count < 3
                        or len(
                            np.unique(
                                np.round(points, decimals=8),
                                axis=0,
                            )
                        )
                        < 3
                        or _contour_area_mm2(points) <= 1e-6
                    )
                    degenerate_count += int(is_degenerate)
                    for image in getattr(
                        contour,
                        "ContourImageSequence",
                        [],
                    ):
                        contour_references_t2 &= (
                            str(image.ReferencedSOPInstanceUID) in t2_sops
                        )

            loaded = RTStructBuilder.create_from(
                dicom_series_path=patched_path,
                rt_struct_path=str(rtstruct_path),
            )
            actual_names = set(loaded.get_roi_names())
            if actual_names != set(expected_rois):
                errors.append(f"declared_roi_mismatch:{rtstruct_path.name}")
                continue
            for roi_name, expected_mask in expected_rois.items():
                actual_mask = loaded.get_roi_mask_by_name(roi_name)
                different_voxels += int(
                    np.logical_xor(actual_mask, expected_mask).sum()
                )

        if not all_xor:
            errors.append("non_xor_contour")
        if degenerate_count:
            errors.append("degenerate_contour")
        if different_voxels:
            errors.append("roundtrip_mask_mismatch")
        if not studies_match:
            errors.append("study_uid_mismatch")
        if not full_t2_reference:
            errors.append("incomplete_t2_reference")
        if not contour_references_t2:
            errors.append("invalid_contour_image_reference")
        if invalid_short_strings:
            errors.append("invalid_short_string")

        return {
            "patient_id": case_directory.name,
            "valid": str(not errors).lower(),
            "rtstruct_count": str(
                sum(path.is_file() for path, _ in rtstruct_specs)
            ),
            "contour_count": str(contour_count),
            "all_closedplanar_xor": str(all_xor).lower(),
            "minimum_contour_points": str(minimum_points or 0),
            "degenerate_contour_count": str(degenerate_count),
            "roundtrip_exact": str(different_voxels == 0).lower(),
            "roundtrip_different_voxels": str(different_voxels),
            "study_matches_t2": str(studies_match).lower(),
            "all_t2_images_referenced": str(full_t2_reference).lower(),
            "contour_images_reference_t2": str(
                contour_references_t2
            ).lower(),
            "invalid_short_string_count": str(invalid_short_strings),
            "errors": ";".join(errors),
        }
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    cases = sorted(
        path.parent
        for path in args.output_root.glob("*/lesions.txt")
    )
    rows = [validate_case(case) for case in cases]
    csv_path = args.csv or args.output_root / "rtstruct_validation.csv"
    json_path = args.json or args.output_root / "rtstruct_validation.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    failures = [row for row in rows if row["valid"] != "true"]
    summary = {
        "case_count": len(rows),
        "valid_case_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "total_contours": sum(int(row["contour_count"]) for row in rows),
        "minimum_contour_points": min(
            (int(row["minimum_contour_points"]) for row in rows),
            default=0,
        ),
        "total_roundtrip_different_voxels": sum(
            int(row["roundtrip_different_voxels"]) for row in rows
        ),
        "failures": failures,
    }
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
