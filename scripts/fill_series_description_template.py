#!/usr/bin/env python3
"""Fill series description investigation CSV from DICOM headers.

Can be used as:
  1. CLI:  python scripts/fill_series_description_template.py --test-data-root test-data/ ...
  2. Notebook cell:  call fill_series_description(...) directly.
"""

import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pydicom
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "pydicom is required. Install with: pip install pydicom"
    ) from exc


TEMPLATE_COLUMNS = [
    "case_id",
    "series_group_id",
    "series_folder",
    "source_path",
    # UIDs
    "sop_instance_uid",
    "sop_class_uid",
    "series_instance_uid",
    "study_instance_uid",
    "study_id",
    "accession_number",
    "patient_id",
    # Acquisition metadata
    "series_number",
    "series_description",
    "protocol_name",
    "sequence_name",
    "modality",
    "image_type",
    "body_part",
    "acquisition_type",
    "scan_options",
    "echo_time_ms",
    "repetition_time_ms",
    "inversion_time_ms",
    "flip_angle_deg",
    "b_value",
    "contrast_agent",
    "orientation",
    "slice_thickness_mm",
    "pixel_spacing",
    "photometric_interpretation",
    "acquisition_device_processing_description",
    "number_of_frames",
    "instance_number",
    "file_size_bytes",
    # Selection flags
    "selected_T2",
    "selected_ADC",
    "selected_HIGHB",
    "regex_candidate",
    "regex_match_result",
    "notes",
]

DEFAULT_CASES = [
    "ProstateX-0000",
    "ProstateX-0001",
    "sgh_data_patient_1",
    "sgh_data_patient_2",
    "sgh_data_patient_3",
    "ProstateX-0004",
    "sgh_data_patient_4",
]


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def _get_attr(ds: "pydicom.Dataset", name: str) -> str:
    return _stringify(getattr(ds, name, None))


def _find_dicom_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(".dcm"):
                yield os.path.join(dirpath, filename)


def _parse_rules(app_path: str) -> Dict[str, List[Dict[str, object]]]:
    """Parse selection rules from app.py, returning ALL selections per rule."""
    if not os.path.exists(app_path):
        return {}

    with open(app_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    rules: Dict[str, List[Dict[str, object]]] = {}
    for name in ("Rules_T2", "Rules_ADC", "Rules_HIGHB"):
        match = re.search(rf"{name}\s*=\s*\"\"\"(.*?)\"\"\"", content, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        selections = data.get("selections", [])
        if not selections:
            continue
        rules[name] = [sel.get("conditions", {}) for sel in selections]
    return rules


def _parse_image_type(value: str) -> List[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in re.split(r"[\\|,]", value) if part.strip()]


def _match_rule(record: Dict[str, str], conditions: Dict[str, object]) -> bool:
    modality = conditions.get("Modality")
    if modality and record.get("modality") != modality:
        return False

    required_image_types = conditions.get("ImageType", [])
    image_types = set(_parse_image_type(record.get("image_type", "")))
    if required_image_types:
        if not all(item in image_types for item in required_image_types):
            return False

    series_regex = conditions.get("SeriesDescription", "")
    if series_regex:
        series_description = record.get("series_description", "")
        try:
            return re.search(series_regex, series_description, re.IGNORECASE) is not None
        except re.error:
            return False
    return False


def _match_any_selection(
    record: Dict[str, str], conditions_list: List[Dict[str, object]]
) -> bool:
    """Try each selection's conditions in order; return True on first match."""
    return any(_match_rule(record, cond) for cond in conditions_list)


def _extract_series_records(
    test_data_root: str,
    case_id: str,
    rules: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, str]]:
    case_root = os.path.join(test_data_root, case_id)
    if not os.path.isdir(case_root):
        return []

    series_map: Dict[Tuple[str, str], Dict[str, str]] = {}

    for filepath in _find_dicom_files(case_root):
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
        except Exception:
            continue

        series_uid = _get_attr(ds, "SeriesInstanceUID")
        study_uid = _get_attr(ds, "StudyInstanceUID")
        key = (series_uid, study_uid)

        if key in series_map:
            continue

        rel_dir = os.path.relpath(os.path.dirname(filepath), case_root)
        series_folder = "" if rel_dir == "." else rel_dir

        try:
            file_size = os.path.getsize(filepath)
        except OSError:
            file_size = ""

        record = {
            "case_id": case_id,
            "series_group_id": "",
            "series_folder": series_folder,
            "source_path": os.path.relpath(filepath, test_data_root),
            # UIDs
            "sop_instance_uid": _get_attr(ds, "SOPInstanceUID"),
            "sop_class_uid": _get_attr(ds, "SOPClassUID"),
            "series_instance_uid": series_uid,
            "study_instance_uid": study_uid,
            "study_id": _get_attr(ds, "StudyID"),
            "accession_number": _get_attr(ds, "AccessionNumber"),
            "patient_id": _get_attr(ds, "PatientID"),
            # Acquisition metadata
            "series_number": _get_attr(ds, "SeriesNumber"),
            "series_description": _get_attr(ds, "SeriesDescription"),
            "protocol_name": _get_attr(ds, "ProtocolName"),
            "sequence_name": _get_attr(ds, "SequenceName"),
            "modality": _get_attr(ds, "Modality"),
            "image_type": _get_attr(ds, "ImageType"),
            "body_part": _get_attr(ds, "BodyPartExamined"),
            "acquisition_type": _get_attr(ds, "MRAcquisitionType"),
            "scan_options": _get_attr(ds, "ScanOptions"),
            "echo_time_ms": _get_attr(ds, "EchoTime"),
            "repetition_time_ms": _get_attr(ds, "RepetitionTime"),
            "inversion_time_ms": _get_attr(ds, "InversionTime"),
            "flip_angle_deg": _get_attr(ds, "FlipAngle"),
            "b_value": _get_attr(ds, "DiffusionBValue"),
            "contrast_agent": _get_attr(ds, "ContrastBolusAgent"),
            "orientation": _get_attr(ds, "ImageOrientationPatient"),
            "slice_thickness_mm": _get_attr(ds, "SliceThickness"),
            "pixel_spacing": _get_attr(ds, "PixelSpacing"),
            "photometric_interpretation": _get_attr(ds, "PhotometricInterpretation"),
            "acquisition_device_processing_description": _get_attr(
                ds, "AcquisitionDeviceProcessingDescription"
            ),
            "number_of_frames": _get_attr(ds, "NumberOfFrames"),
            "instance_number": _get_attr(ds, "InstanceNumber"),
            "file_size_bytes": _stringify(file_size),
            # Selection flags
            "selected_T2": "",
            "selected_ADC": "",
            "selected_HIGHB": "",
            "regex_candidate": "",
            "regex_match_result": "",
            "notes": "",
        }

        if rules:
            matches = []
            is_t2 = "Rules_T2" in rules and _match_any_selection(record, rules["Rules_T2"])
            is_adc = "Rules_ADC" in rules and _match_any_selection(record, rules["Rules_ADC"])
            is_highb = "Rules_HIGHB" in rules and _match_any_selection(record, rules["Rules_HIGHB"])

            record["selected_T2"] = "1" if is_t2 else "0"
            record["selected_ADC"] = "1" if is_adc else "0"
            record["selected_HIGHB"] = "1" if is_highb else "0"

            if is_t2:
                matches.append("T2")
            if is_adc:
                matches.append("ADC")
            if is_highb:
                matches.append("HIGHB")

            record["regex_candidate"] = record.get("series_description", "")
            record["regex_match_result"] = ",".join(matches)
            if len(matches) > 1:
                record["notes"] = "multiple rule matches"

        series_map[key] = record

    records = list(series_map.values())
    records.sort(key=lambda row: (row.get("series_number", ""), row.get("series_description", "")))
    for idx, record in enumerate(records, start=1):
        record["series_group_id"] = str(idx)

    return records


def _load_template_rows(template_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(template_path):
        return []

    with open(template_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_rows(output_path: str, rows: List[Dict[str, str]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in TEMPLATE_COLUMNS})


def fill_series_description(
    test_data_root: str,
    output_path: str,
    cases: List[str],
    app_path: str,
) -> str:
    """Scan DICOM directories and write a filled CSV with metadata and selection flags.

    This is the notebook-friendly entry point. Returns the output path on success.

    Parameters
    ----------
    test_data_root : str
        Root directory containing one subfolder per case, each with DICOM files.
    output_path : str
        Path for the output CSV file.
    cases : list[str]
        Case folder names to scan inside *test_data_root*.
    app_path : str
        Path to ``app.py`` containing Rules_T2 / Rules_ADC / Rules_HIGHB.
    """
    rules = _parse_rules(app_path)

    records: List[Dict[str, str]] = []
    for case_id in cases:
        case_records = _extract_series_records(test_data_root, case_id, rules)
        records.extend(case_records)
        print(f"  {case_id}: {len(case_records)} series found")

    if not records:
        print("WARNING: No DICOM series found. Check test_data_root and cases.")
        return output_path

    _write_rows(output_path, records)
    print(f"\nWrote {len(records)} series to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill series description investigation template from DICOM headers."
    )
    parser.add_argument(
        "--test-data-root",
        default=os.path.join(os.getcwd(), "test-data"),
        help="Path to the test-data directory.",
    )
    parser.add_argument(
        "--template",
        default=os.path.join(os.getcwd(), "series_description_investigation_template.csv"),
        help="Path to the CSV template.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.getcwd(), "series_description_investigation_filled.csv"),
        help="Path to the output CSV.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        help="Case IDs to scan.",
    )
    parser.add_argument(
        "--app-path",
        default=os.path.join(os.getcwd(), "prostate_mri_lesion_seg_app", "app.py"),
        help="Path to app.py with selection rules.",
    )
    args = parser.parse_args()

    fill_series_description(
        test_data_root=args.test_data_root,
        output_path=args.output,
        cases=args.cases,
        app_path=args.app_path,
    )

    template_rows = _load_template_rows(args.template)
    if template_rows:
        print(
            "Existing template rows are preserved in "
            "series_description_investigation_template.csv."
        )


if __name__ == "__main__":
    main()
