from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import binary_fill_holes, label as ndi_label


AUDIT_COLUMNS = {
    "case": "Case",
    "reviewed": "Reviewed",
    "q1": "Q1: Prostate contour acceptable",
    "q2": "Q2: All suspicious lesions contoured",
    "q3": "Q3: No incorrectly contoured regions",
    "q4": "Q4: Include in training set",
    "q5": "Q5: Missed obvious tumor",
}

QUESTION_COLUMNS = {
    "Q1": AUDIT_COLUMNS["q1"],
    "Q2": AUDIT_COLUMNS["q2"],
    "Q3": AUDIT_COLUMNS["q3"],
    "Q4": AUDIT_COLUMNS["q4"],
    "Q5": AUDIT_COLUMNS["q5"],
}

MANIFEST_COLUMNS = [
    "subject_id",
    "source",
    "reviewed",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "organ_eligible",
    "lesion_eligible",
    "organ_exclusion_reason",
    "lesion_exclusion_reason",
    "t2_path",
    "adc_path",
    "highb_path",
    "organ_label_path",
    "lesion_label_path",
    "organ_foreground_voxels",
    "lesion_foreground_voxels",
    "lesion_is_empty",
    "manufacturer_raw",
    "manufacturer",
    "model",
    "study_date",
    "study_year",
]


@dataclass(frozen=True)
class DicomMetadata:
    manufacturer_raw: str = ""
    manufacturer: str = "UNKNOWN"
    model: str = "UNKNOWN"
    study_date: str = ""
    study_year: str = "UNKNOWN"


def _answer(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"Y", "N", "P"}:
        return text
    return ""


def _as_bool_text(value: bool) -> str:
    return "true" if value else "false"


def _bool_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def normalize_manufacturer(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).upper()
    if not text:
        return "UNKNOWN"
    if "SIEMENS" in text:
        return "SIEMENS"
    if "PHILIPS" in text:
        return "PHILIPS"
    if text in {"GE", "GEMS"} or "GENERAL ELECTRIC" in text or text.startswith("GE "):
        return "GE"
    if "CANON" in text or "TOSHIBA" in text:
        return "CANON/TOSHIBA"
    return text


def read_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in AUDIT_COLUMNS.values() if column not in fieldnames]
        if missing:
            raise ValueError(f"Audit CSV is missing required column(s): {', '.join(missing)}")
        rows = [dict(row) for row in reader]

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        subject_id = str(row[AUDIT_COLUMNS["case"]] or "").strip()
        if not subject_id:
            raise ValueError("Audit CSV contains a row with a blank Case value")
        if subject_id in seen:
            duplicates.append(subject_id)
        seen.add(subject_id)
    if duplicates:
        raise ValueError(
            "Audit CSV contains duplicate Case value(s): "
            + ", ".join(sorted(set(duplicates))[:10])
        )
    return rows


def audit_eligibility(row: Mapping[str, object]) -> tuple[bool, bool]:
    reviewed = _answer(row.get(AUDIT_COLUMNS["reviewed"])) == "Y"
    organ = (
        reviewed
        and _answer(row.get(AUDIT_COLUMNS["q1"])) == "Y"
        and _answer(row.get(AUDIT_COLUMNS["q4"])) == "Y"
    )
    lesion = (
        reviewed
        and _answer(row.get(AUDIT_COLUMNS["q2"])) == "Y"
        and _answer(row.get(AUDIT_COLUMNS["q3"])) == "Y"
        and _answer(row.get(AUDIT_COLUMNS["q4"])) == "Y"
    )
    return organ, lesion


def clean_binary_organ_mask(mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the inference organ cleanup policy and return a binary label."""
    data = np.rint(np.asarray(mask)).astype(np.uint8)
    binary = data > 0
    foreground = int(binary.sum())
    metrics: dict[str, object] = {
        "original_component_count": 0,
        "original_foreground_voxels": foreground,
        "largest_component_voxels": 0,
        "removed_island_count": 0,
        "removed_voxels": 0,
        "filled_hole_voxels": 0,
        "cleaned_foreground_voxels": 0,
    }
    if foreground == 0:
        return binary.astype(np.uint8), metrics

    structure = np.ones((3,) * binary.ndim, dtype=np.uint8)
    components, count = ndi_label(binary, structure=structure)
    sizes = np.bincount(components.ravel())
    largest_id = int(np.argmax(sizes[1:]) + 1)
    largest = components == largest_id
    filled = binary_fill_holes(largest)

    metrics.update(
        {
            "original_component_count": int(count),
            "largest_component_voxels": int(largest.sum()),
            "removed_island_count": max(int(count) - 1, 0),
            "removed_voxels": foreground - int(largest.sum()),
            "filled_hole_voxels": int((filled & ~largest).sum()),
            "cleaned_foreground_voxels": int(filled.sum()),
        }
    )
    return filled.astype(np.uint8), metrics


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _prepare_clean_organ_label(
    source_path: Path,
    *,
    subject_id: str,
    cleaned_dir: Path,
) -> tuple[Path, int]:
    image = nib.load(str(source_path))
    cleaned, _ = clean_binary_organ_mask(np.asanyarray(image.dataobj))
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    destination = cleaned_dir / f"{_safe_identifier(subject_id)}.nii.gz"

    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(cleaned, image.affine, header=header), str(destination))
    return destination.resolve(), int(cleaned.sum())


def _mask_info(path: Path, *, binary: bool = True) -> tuple[int, set[int]]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError("contains non-finite values")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, atol=1e-4):
        raise ValueError("contains non-integer label values")
    values = {int(value) for value in np.unique(rounded)}
    if binary and not values.issubset({0, 1}):
        raise ValueError(f"contains unexpected labels {sorted(values)}")
    return int(np.count_nonzero(rounded)), values


def _validate_nifti_headers(paths: Sequence[tuple[str, Path]]) -> str:
    for label, path in paths:
        try:
            image = nib.load(str(path))
            if image.ndim not in {2, 3}:
                return f"invalid {label}: expected 2D/3D NIfTI, got {image.shape}"
            if any(int(size) < 1 for size in image.shape):
                return f"invalid {label}: empty dimension in {image.shape}"
            zooms = np.asarray(image.header.get_zooms()[: image.ndim], dtype=float)
            if not np.all(np.isfinite(zooms)) or np.any(zooms <= 0):
                return f"invalid {label}: non-positive or non-finite spacing"
        except Exception as exc:
            return f"invalid {label}: {exc}"
    return ""


def _find_first_dicom(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    preferred = directory / "dicom" / "t2"
    search_roots = [preferred, directory] if preferred.is_dir() else [directory]
    for root in search_roots:
        for candidate in sorted(root.rglob("*")):
            if candidate.is_file() and (
                candidate.suffix.lower() in {".dcm", ".dicom"} or not candidate.suffix
            ):
                try:
                    dataset = pydicom.dcmread(
                        str(candidate),
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=[
                            "SOPInstanceUID",
                            "Modality",
                            "Manufacturer",
                            "ManufacturerModelName",
                            "StudyDate",
                        ],
                    )
                    if getattr(dataset, "SOPInstanceUID", None) or getattr(
                        dataset, "Modality", None
                    ):
                        return candidate
                except Exception:
                    continue
    return None


def read_dicom_metadata(case_output_dir: Path, dicom_case_dir: Path | None = None) -> DicomMetadata:
    first = _find_first_dicom(case_output_dir)
    if first is None and dicom_case_dir is not None:
        first = _find_first_dicom(dicom_case_dir)
    if first is None:
        return DicomMetadata()

    try:
        dataset = pydicom.dcmread(
            str(first),
            stop_before_pixels=True,
            force=True,
            specific_tags=["Manufacturer", "ManufacturerModelName", "StudyDate"],
        )
    except Exception:
        return DicomMetadata()

    manufacturer_raw = str(getattr(dataset, "Manufacturer", "") or "").strip()
    model = str(getattr(dataset, "ManufacturerModelName", "") or "").strip() or "UNKNOWN"
    study_date = str(getattr(dataset, "StudyDate", "") or "").strip()
    study_year = study_date[:4] if re.fullmatch(r"\d{8}", study_date) else "UNKNOWN"
    return DicomMetadata(
        manufacturer_raw=manufacturer_raw,
        manufacturer=normalize_manufacturer(manufacturer_raw),
        model=model,
        study_date=study_date,
        study_year=study_year,
    )


def _existing(path: Path) -> str:
    return str(path.resolve()) if path.is_file() else ""


def _technical_reason(required_paths: Sequence[tuple[str, Path]]) -> str:
    missing = [label for label, path in required_paths if not path.is_file()]
    return "" if not missing else "missing " + ", ".join(missing)


def _sgh_manifest_row(
    audit_row: Mapping[str, object],
    *,
    output_root: Path,
    dicom_root: Path | None,
    cleaned_dir: Path,
) -> dict[str, str]:
    subject_id = str(audit_row[AUDIT_COLUMNS["case"]] or "").strip()
    case_dir = output_root / subject_id
    t2 = case_dir / "t2" / "t2.nii.gz"
    adc = case_dir / "adc" / "adc.nii.gz"
    highb = case_dir / "highb" / "highb.nii.gz"
    raw_organ = case_dir / "organ" / "organ.nii.gz"
    cleaned_organ = case_dir / "organ" / "cleaned_organ.nii.gz"
    lesion = case_dir / "lesion" / "lesion_mask.nii.gz"

    organ_policy, lesion_policy = audit_eligibility(audit_row)
    organ_reason = "" if organ_policy else "audit policy"
    lesion_reason = "" if lesion_policy else "audit policy"
    organ_foreground = ""
    lesion_foreground = ""
    lesion_is_empty = ""
    organ_label_path = ""

    base_required = [("T2", t2)]
    if organ_policy:
        reason = _technical_reason(base_required + [("organ mask", cleaned_organ if cleaned_organ.is_file() else raw_organ)])
        if not reason:
            reason = _validate_nifti_headers([("T2", t2)])
        if reason:
            organ_reason = reason
        else:
            try:
                source = cleaned_organ if cleaned_organ.is_file() else raw_organ
                prepared_path, foreground = _prepare_clean_organ_label(
                    source,
                    subject_id=f"SGH_{subject_id}",
                    cleaned_dir=cleaned_dir,
                )
                if foreground == 0:
                    organ_reason = "empty cleaned organ mask"
                else:
                    organ_label_path = str(prepared_path)
                    organ_foreground = str(foreground)
            except Exception as exc:
                organ_reason = f"invalid organ mask: {exc}"

    if lesion_policy:
        reason = _technical_reason(
            [
                ("T2", t2),
                ("ADC", adc),
                ("HIGHB", highb),
                ("cleaned organ mask", cleaned_organ if cleaned_organ.is_file() else raw_organ),
                ("lesion mask", lesion),
            ]
        )
        if reason:
            lesion_reason = reason
        else:
            lesion_reason = _validate_nifti_headers(
                [("T2", t2), ("ADC", adc), ("HIGHB", highb)]
            )
            if not lesion_reason:
                try:
                    foreground, _ = _mask_info(lesion, binary=True)
                    lesion_foreground = str(foreground)
                    lesion_is_empty = _as_bool_text(foreground == 0)
                    if not organ_label_path:
                        source = cleaned_organ if cleaned_organ.is_file() else raw_organ
                        prepared_path, prepared_foreground = _prepare_clean_organ_label(
                            source,
                            subject_id=f"SGH_{subject_id}",
                            cleaned_dir=cleaned_dir,
                        )
                        if prepared_foreground == 0:
                            lesion_reason = "empty cleaned organ mask"
                        else:
                            organ_label_path = str(prepared_path)
                            organ_foreground = str(prepared_foreground)
                except Exception as exc:
                    lesion_reason = f"invalid lesion/organ mask: {exc}"

    metadata = read_dicom_metadata(
        case_dir,
        (dicom_root / subject_id) if dicom_root is not None else None,
    )
    values = {
        key: _answer(audit_row.get(column))
        for key, column in (
            ("reviewed", AUDIT_COLUMNS["reviewed"]),
            ("q1", AUDIT_COLUMNS["q1"]),
            ("q2", AUDIT_COLUMNS["q2"]),
            ("q3", AUDIT_COLUMNS["q3"]),
            ("q4", AUDIT_COLUMNS["q4"]),
            ("q5", AUDIT_COLUMNS["q5"]),
        )
    }
    return {
        "subject_id": subject_id,
        "source": "SGH",
        **values,
        "organ_eligible": _as_bool_text(organ_policy and not organ_reason),
        "lesion_eligible": _as_bool_text(lesion_policy and not lesion_reason),
        "organ_exclusion_reason": organ_reason,
        "lesion_exclusion_reason": lesion_reason,
        "t2_path": _existing(t2),
        "adc_path": _existing(adc),
        "highb_path": _existing(highb),
        "organ_label_path": organ_label_path,
        "lesion_label_path": _existing(lesion),
        "organ_foreground_voxels": organ_foreground,
        "lesion_foreground_voxels": lesion_foreground,
        "lesion_is_empty": lesion_is_empty,
        **asdict(metadata),
    }


def _subject_stem(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def _prostatex_manifest_rows(
    root: Path,
    *,
    cleaned_dir: Path,
) -> Iterator[dict[str, str]]:
    t2_dir = root / "t2"
    for t2 in sorted(t2_dir.glob("*.nii.gz")):
        subject_id = _subject_stem(t2, "_t2")
        adc = root / "adc" / f"{subject_id}_adc.nii.gz"
        highb = root / "highb" / f"{subject_id}_highb.nii.gz"
        organ = root / "organ_masks" / f"{subject_id}.nii.gz"
        lesion = root / "lesion_masks" / f"{subject_id}.nii.gz"

        organ_reason = _technical_reason([("T2", t2), ("organ mask", organ)])
        lesion_reason = _technical_reason(
            [
                ("T2", t2),
                ("ADC", adc),
                ("HIGHB", highb),
                ("organ mask", organ),
                ("lesion mask", lesion),
            ]
        )
        organ_label_path = ""
        organ_foreground = ""
        lesion_foreground = ""
        lesion_is_empty = ""
        if not organ_reason:
            organ_reason = _validate_nifti_headers([("T2", t2)])
        if not organ_reason:
            try:
                prepared_path, foreground = _prepare_clean_organ_label(
                    organ,
                    subject_id=f"ProstateX_{subject_id}",
                    cleaned_dir=cleaned_dir,
                )
                if foreground == 0:
                    organ_reason = "empty cleaned organ mask"
                else:
                    organ_label_path = str(prepared_path)
                    organ_foreground = str(foreground)
            except Exception as exc:
                organ_reason = f"invalid organ mask: {exc}"
        if not lesion_reason:
            lesion_reason = _validate_nifti_headers(
                [("T2", t2), ("ADC", adc), ("HIGHB", highb)]
            )
        if not lesion_reason:
            try:
                foreground, _ = _mask_info(lesion, binary=True)
                lesion_foreground = str(foreground)
                lesion_is_empty = _as_bool_text(foreground == 0)
            except Exception as exc:
                lesion_reason = f"invalid lesion mask: {exc}"

        yield {
            "subject_id": subject_id,
            "source": "ProstateX",
            "reviewed": "",
            "q1": "",
            "q2": "",
            "q3": "",
            "q4": "",
            "q5": "",
            "organ_eligible": _as_bool_text(not organ_reason),
            "lesion_eligible": _as_bool_text(not lesion_reason),
            "organ_exclusion_reason": organ_reason,
            "lesion_exclusion_reason": lesion_reason,
            "t2_path": _existing(t2),
            "adc_path": _existing(adc),
            "highb_path": _existing(highb),
            "organ_label_path": organ_label_path,
            "lesion_label_path": _existing(lesion),
            "organ_foreground_voxels": organ_foreground,
            "lesion_foreground_voxels": lesion_foreground,
            "lesion_is_empty": lesion_is_empty,
            "manufacturer_raw": "",
            "manufacturer": "PUBLIC/UNKNOWN",
            "model": "PUBLIC/UNKNOWN",
            "study_date": "",
            "study_year": "UNKNOWN",
        }


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MANIFEST_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing column(s): {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    identities = [(row["source"], row["subject_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Manifest contains duplicate source/subject_id pairs")
    return rows


def _question_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    reviewed_rows = [
        row for row in rows if _answer(row.get(AUDIT_COLUMNS["reviewed"])) == "Y"
    ]
    output: list[dict[str, object]] = []
    for question, column in QUESTION_COLUMNS.items():
        counts = Counter(_answer(row.get(column)) or "BLANK" for row in reviewed_rows)
        answered = counts["Y"] + counts["N"]
        output.append(
            {
                "question": question,
                "description": column,
                "denominator_reviewed": len(reviewed_rows),
                "yes": counts["Y"],
                "no": counts["N"],
                "pending": counts["P"],
                "blank": counts["BLANK"],
                "yes_percent_reviewed": round(
                    100.0 * counts["Y"] / len(reviewed_rows), 2
                )
                if reviewed_rows
                else 0.0,
                "yes_percent_answered": round(100.0 * counts["Y"] / answered, 2)
                if answered
                else 0.0,
            }
        )
    return output


def _breakdown_rows(
    audit_rows: Sequence[Mapping[str, object]],
    manifest_by_id: Mapping[str, Mapping[str, str]],
    *,
    group_field: str,
) -> list[dict[str, object]]:
    reviewed = [
        row for row in audit_rows if _answer(row.get(AUDIT_COLUMNS["reviewed"])) == "Y"
    ]
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in reviewed:
        subject_id = str(row[AUDIT_COLUMNS["case"]] or "").strip()
        group_value = manifest_by_id.get(subject_id, {}).get(group_field, "") or "UNKNOWN"
        grouped[group_value].append(row)

    output: list[dict[str, object]] = []
    for group_value in sorted(grouped):
        group_rows = grouped[group_value]
        for question, column in QUESTION_COLUMNS.items():
            counts = Counter(_answer(row.get(column)) or "BLANK" for row in group_rows)
            answered = counts["Y"] + counts["N"]
            output.append(
                {
                    "group_field": group_field,
                    "group_value": group_value,
                    "question": question,
                    "reviewed_cases": len(group_rows),
                    "yes": counts["Y"],
                    "no": counts["N"],
                    "blank": counts["BLANK"],
                    "yes_percent_reviewed": round(
                        100.0 * counts["Y"] / len(group_rows), 2
                    )
                    if group_rows
                    else 0.0,
                    "yes_percent_answered": round(100.0 * counts["Y"] / answered, 2)
                    if answered
                    else 0.0,
                }
            )
    return output


def _hash_fraction(seed: int, subject_key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{subject_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _split_stratum(row: Mapping[str, str], task: str) -> tuple[str, ...]:
    manufacturer = row.get("manufacturer", "UNKNOWN") or "UNKNOWN"
    year = row.get("study_year", "UNKNOWN") or "UNKNOWN"
    if year.isdigit():
        year_group = f"{int(year) // 5 * 5}-{int(year) // 5 * 5 + 4}"
    else:
        year_group = "UNKNOWN"
    result = [row.get("source", "UNKNOWN"), manufacturer, year_group]
    if task == "lesion":
        result.append(row.get("lesion_is_empty", ""))
    return tuple(result)


def create_splits(
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    task: str,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
    folds: int = 5,
) -> list[dict[str, str]]:
    if task not in {"organ", "lesion"}:
        raise ValueError("task must be 'organ' or 'lesion'")
    eligible_column = f"{task}_eligible"
    eligible = [dict(row) for row in manifest_rows if _bool_value(row.get(eligible_column))]
    if len(eligible) < 3:
        raise ValueError(f"Need at least three eligible {task} cases to create splits")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        grouped[_split_stratum(row, task)].append(row)

    output: list[dict[str, str]] = []
    for stratum, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: _hash_fraction(
                seed, f"{row['source']}:{row['subject_id']}"
            ),
        )
        n = len(ordered)
        n_test = min(n, max(0, int(round(n * test_fraction))))
        test_rows = ordered[:n_test]
        development = ordered[n_test:]

        test_ids = {(row["source"], row["subject_id"]) for row in test_rows}
        if task == "organ":
            n_val = min(
                len(development),
                max(0, int(round(n * validation_fraction))),
            )
            val_ids = {
                (row["source"], row["subject_id"])
                for row in development[:n_val]
            }
            for row in ordered:
                identity = (row["source"], row["subject_id"])
                split = "test" if identity in test_ids else ("val" if identity in val_ids else "train")
                output.append(
                    {
                        "source": row["source"],
                        "subject_id": row["subject_id"],
                        "split": split,
                        "fold": "",
                        "stratum": "|".join(stratum),
                    }
                )
        else:
            for dev_index, row in enumerate(development):
                row["_assigned_fold"] = str(dev_index % folds)
            for row in ordered:
                identity = (row["source"], row["subject_id"])
                output.append(
                    {
                        "source": row["source"],
                        "subject_id": row["subject_id"],
                        "split": "test" if identity in test_ids else "development",
                        "fold": "" if identity in test_ids else row["_assigned_fold"],
                        "stratum": "|".join(stratum),
                    }
                )

    output.sort(key=lambda row: (row["source"], row["subject_id"]))
    identities = [(row["source"], row["subject_id"]) for row in output]
    if len(identities) != len(set(identities)):
        raise AssertionError("Split generation produced duplicate subjects")
    if task == "organ":
        present = {row["split"] for row in output}
        if present != {"train", "val", "test"}:
            raise ValueError(
                "Organ cohort is too small/fragmented to produce non-empty "
                "train, validation, and test partitions"
            )
    else:
        present_folds = {
            row["fold"] for row in output if row["split"] == "development"
        }
        expected_folds = {str(fold) for fold in range(folds)}
        if present_folds != expected_folds:
            raise ValueError(
                "Lesion development cohort cannot populate all five folds; "
                "add eligible cases (normally ProstateX + SGH)"
            )
        if not any(row["split"] == "test" for row in output):
            raise ValueError("Lesion split has no held-out test cases")
    return output


def _write_split_files(
    output_dir: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> None:
    split_columns = ["source", "subject_id", "split", "fold", "stratum"]
    organ = create_splits(
        manifest_rows,
        task="organ",
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    lesion = create_splits(
        manifest_rows,
        task="lesion",
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        folds=5,
    )
    write_csv(output_dir / "organ_splits.csv", organ, split_columns)
    write_csv(output_dir / "lesion_splits.csv", lesion, split_columns)

    lesion_lookup = {
        (row["source"], row["subject_id"]): row for row in lesion
    }
    for fold in range(5):
        train_rows: list[dict[str, str]] = []
        val_rows: list[dict[str, str]] = []
        for manifest_row in manifest_rows:
            identity = (manifest_row["source"], manifest_row["subject_id"])
            assignment = lesion_lookup.get(identity)
            if not assignment or assignment["split"] == "test":
                continue
            destination = val_rows if assignment["fold"] == str(fold) else train_rows
            destination.append(
                {
                    "subject_id": manifest_row["subject_id"],
                    "source": manifest_row["source"],
                }
            )
        write_csv(
            output_dir / f"fold{fold}_train.csv",
            train_rows,
            ["subject_id", "source"],
        )
        write_csv(
            output_dir / f"fold{fold}_val.csv",
            val_rows,
            ["subject_id", "source"],
        )


def _write_report(
    output_dir: Path,
    *,
    audit_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, str]],
    question_summary: Sequence[Mapping[str, object]],
) -> None:
    reviewed_counts = Counter(
        _answer(row.get(AUDIT_COLUMNS["reviewed"])) or "BLANK" for row in audit_rows
    )
    organ_count = sum(_bool_value(row["organ_eligible"]) for row in manifest_rows)
    lesion_count = sum(_bool_value(row["lesion_eligible"]) for row in manifest_rows)
    exclusions = Counter()
    for row in manifest_rows:
        if not _bool_value(row["organ_eligible"]):
            exclusions[f"organ: {row['organ_exclusion_reason'] or 'not eligible'}"] += 1
        if not _bool_value(row["lesion_eligible"]):
            exclusions[f"lesion: {row['lesion_exclusion_reason'] or 'not eligible'}"] += 1

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "assigned_cases": len(audit_rows),
        "reviewed_counts": dict(reviewed_counts),
        "manifest_cases": len(manifest_rows),
        "organ_eligible": organ_count,
        "lesion_eligible": lesion_count,
        "question_summary": list(question_summary),
        "exclusions": dict(exclusions),
    }
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    lines = [
        "# Prostate MRI audit and training-cohort report",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"- Assigned SGH cases: {len(audit_rows):,}",
        f"- Reviewed Y: {reviewed_counts['Y']:,}",
        f"- Reviewed P: {reviewed_counts['P']:,}",
        f"- Reviewed blank: {reviewed_counts['BLANK']:,}",
        f"- Organ-eligible cases after audit and technical QC: {organ_count:,}",
        f"- Lesion-eligible cases after audit and technical QC: {lesion_count:,}",
        "",
        "## Question summary among Reviewed=Y",
        "",
        "| Question | Yes | No | Blank | Yes % reviewed | Yes % answered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in question_summary:
        lines.append(
            f"| {row['question']} | {row['yes']:,} | {row['no']:,} | "
            f"{row['blank']:,} | {row['yes_percent_reviewed']:.2f}% | "
            f"{row['yes_percent_answered']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Percentages always retain their denominator. Blank Q5 responses are unknown, not No.",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_cohort(
    *,
    audit_csv: Path,
    sgh_output_root: Path,
    output_dir: Path,
    prostatex_root: Path | None = None,
    dicom_root: Path | None = None,
    seed: int = 42,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> Path:
    audit_rows = read_audit_rows(audit_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = output_dir / "prepared_organ_labels"

    manifest_rows = [
        _sgh_manifest_row(
            row,
            output_root=sgh_output_root,
            dicom_root=dicom_root,
            cleaned_dir=cleaned_dir,
        )
        for row in audit_rows
    ]
    if prostatex_root is not None:
        manifest_rows.extend(
            _prostatex_manifest_rows(prostatex_root, cleaned_dir=cleaned_dir)
        )
    manifest_rows.sort(key=lambda row: (row["source"], row["subject_id"]))

    manifest_path = output_dir / "training_manifest.csv"
    write_csv(manifest_path, manifest_rows, MANIFEST_COLUMNS)

    question_summary = _question_summary(audit_rows)
    write_csv(
        output_dir / "audit_question_summary.csv",
        question_summary,
        [
            "question",
            "description",
            "denominator_reviewed",
            "yes",
            "no",
            "pending",
            "blank",
            "yes_percent_reviewed",
            "yes_percent_answered",
        ],
    )
    by_id = {
        row["subject_id"]: row for row in manifest_rows if row["source"] == "SGH"
    }
    breakdown_columns = [
        "group_field",
        "group_value",
        "question",
        "reviewed_cases",
        "yes",
        "no",
        "blank",
        "yes_percent_reviewed",
        "yes_percent_answered",
    ]
    for field, filename in (
        ("manufacturer_raw", "audit_by_manufacturer_raw.csv"),
        ("manufacturer", "audit_by_manufacturer.csv"),
        ("model", "audit_by_model.csv"),
        ("study_date", "audit_by_study_date.csv"),
        ("study_year", "audit_by_study_year.csv"),
    ):
        write_csv(
            output_dir / filename,
            _breakdown_rows(audit_rows, by_id, group_field=field),
            breakdown_columns,
        )

    exclusions = [
        row
        for row in manifest_rows
        if not _bool_value(row["organ_eligible"])
        or not _bool_value(row["lesion_eligible"])
    ]
    write_csv(output_dir / "exclusions.csv", exclusions, MANIFEST_COLUMNS)
    _write_split_files(
        output_dir,
        manifest_rows,
        seed=seed,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    _write_report(
        output_dir,
        audit_rows=audit_rows,
        manifest_rows=manifest_rows,
        question_summary=question_summary,
    )
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reviewed SGH + ProstateX training manifests, reports, and splits."
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--sgh-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prostatex-root", type=Path)
    parser.add_argument("--dicom-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.audit_csv.is_file():
        raise FileNotFoundError(args.audit_csv)
    if not args.sgh_output_root.is_dir():
        raise NotADirectoryError(args.sgh_output_root)
    if args.prostatex_root is not None and not args.prostatex_root.is_dir():
        raise NotADirectoryError(args.prostatex_root)
    manifest = prepare_cohort(
        audit_csv=args.audit_csv,
        sgh_output_root=args.sgh_output_root,
        output_dir=args.output_dir,
        prostatex_root=args.prostatex_root,
        dicom_root=args.dicom_root,
        seed=args.seed,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )
    rows = read_manifest(manifest)
    print(f"Manifest: {manifest}")
    print(f"Cases: {len(rows)}")
    print(f"Organ eligible: {sum(_bool_value(row['organ_eligible']) for row in rows)}")
    print(f"Lesion eligible: {sum(_bool_value(row['lesion_eligible']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
