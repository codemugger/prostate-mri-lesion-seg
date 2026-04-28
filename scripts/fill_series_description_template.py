#!/usr/bin/env python3
"""Fill series description investigation CSV from DICOM headers.

Supports:
  - Exhaustive b-value enumeration across all slices per series
  - Resume from interrupted runs (automatically skips completed cases)
  - Multi-threaded case processing for large datasets over slow I/O
  - Graceful Ctrl+C — progress is saved, re-run to continue

Usage:
  CLI:
    python scripts/fill_series_description_template.py \\
        --test-data-root "/mnt/d/Anson/sgh de-id data comprehensive" \\
        --output series_description_filled.csv \\
        --num-workers 8

  Notebook:
    from scripts.fill_series_description_template import fill_series_description
    fill_series_description(
        test_data_root="/mnt/d/Anson/sgh de-id data comprehensive",
        output_path="series_description_filled.csv",
        app_path="prostate_mri_lesion_seg_app/app.py",
    )
"""

import argparse
import ast
import csv
import json
import os
import re
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import pydicom
except ImportError as exc:
    raise SystemExit(
        "pydicom is required. Install with: pip install pydicom"
    ) from exc

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Output CSV columns — image_type is intentionally excluded from the CSV but
# still extracted internally because _match_rule() needs it for ImageType
# conditions in the selection rules.
# ---------------------------------------------------------------------------
TEMPLATE_COLUMNS = [
    "case_id",
    "series_group_id",
    "series_folder",
    "source_path",
    # UIDs
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
    "body_part",
    "acquisition_type",
    "scan_options",
    "echo_time_ms",
    "repetition_time_ms",
    "inversion_time_ms",
    "flip_angle_deg",
    "b_value",
    "slice_thickness_mm",
    "instance_number",
    "file_size_bytes",
    # Exhaustive per-series aggregation
    "total_slices",
    "unique_b_values",
    "slices_per_b_value",
    # Selection flags
    "selected_T2",
    "selected_ADC",
    "selected_HIGHB",
    "regex_candidate",
    "regex_match_result",
    "notes",
]

_SIEMENS_B_VALUE_TAG = (0x0019, 0x100C)

SPECIFIC_TAGS = [
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "StudyID",
    "AccessionNumber",
    "PatientID",
    "SeriesNumber",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "Modality",
    "ImageType",
    "BodyPartExamined",
    "MRAcquisitionType",
    "ScanOptions",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "FlipAngle",
    "DiffusionBValue",
    "SliceThickness",
    "InstanceNumber",
    _SIEMENS_B_VALUE_TAG,
]

_shutdown = threading.Event()
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        for filename in sorted(filenames):
            if filename.lower().endswith(".dcm"):
                yield os.path.join(dirpath, filename)


def _normalize_b_value(raw: object) -> str:
    """Coerce a raw b-value (float, int, bytes, str) to a clean numeric string."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", errors="ignore").strip()
    raw = str(raw).strip()
    if not raw:
        return ""
    try:
        f = float(raw)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return raw


def _get_b_value(ds: "pydicom.Dataset") -> str:
    """Extract b-value: standard tag (0018,9087) first, Siemens (0019,100c) fallback."""
    standard = getattr(ds, "DiffusionBValue", None)
    if standard is not None:
        return _normalize_b_value(standard)
    siemens_elem = ds.get(_SIEMENS_B_VALUE_TAG)
    if siemens_elem is not None:
        return _normalize_b_value(siemens_elem.value)
    return ""


# ---------------------------------------------------------------------------
# Rule parsing and matching (unchanged logic, used for selection flags)
# ---------------------------------------------------------------------------

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
    return any(_match_rule(record, cond) for cond in conditions_list)


# ---------------------------------------------------------------------------
# Core scanning — now reads ALL slices to aggregate b-values per series
# ---------------------------------------------------------------------------

def _sort_b_values(b_values: Iterable[str]) -> List[str]:
    """Deduplicate and sort b-value strings numerically."""
    def _to_float(v: str) -> float:
        try:
            return float(v)
        except (ValueError, TypeError):
            return float("inf")
    return sorted(set(b_values), key=_to_float)


def _extract_series_records(
    test_data_root: str,
    case_id: str,
    rules: Dict[str, List[Dict[str, object]]],
) -> List[Dict[str, str]]:
    """Scan ALL DICOMs in a case, aggregating b-values and slice counts per series."""
    case_root = os.path.join(test_data_root, case_id)
    if not os.path.isdir(case_root):
        return []

    series_meta: Dict[Tuple[str, str], Dict[str, str]] = {}
    series_b_values: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    series_slice_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for filepath in _find_dicom_files(case_root):
        if _shutdown.is_set():
            return []

        try:
            ds = pydicom.dcmread(
                filepath,
                stop_before_pixels=True,
                specific_tags=SPECIFIC_TAGS,
                force=True,
            )
        except Exception:
            continue

        series_uid = _get_attr(ds, "SeriesInstanceUID")
        study_uid = _get_attr(ds, "StudyInstanceUID")
        key = (series_uid, study_uid)

        series_slice_counts[key] += 1
        b_val = _get_b_value(ds)
        if b_val:
            series_b_values[key].append(b_val)

        if key in series_meta:
            continue

        # First file in this series — capture full metadata
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
            "series_instance_uid": series_uid,
            "study_instance_uid": study_uid,
            "study_id": _get_attr(ds, "StudyID"),
            "accession_number": _get_attr(ds, "AccessionNumber"),
            "patient_id": _get_attr(ds, "PatientID"),
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
            "b_value": b_val,
            "slice_thickness_mm": _get_attr(ds, "SliceThickness"),
            "instance_number": _get_attr(ds, "InstanceNumber"),
            "file_size_bytes": _stringify(file_size),
            "total_slices": "",
            "unique_b_values": "",
            "slices_per_b_value": "",
            "selected_T2": "",
            "selected_ADC": "",
            "selected_HIGHB": "",
            "regex_candidate": "",
            "regex_match_result": "",
            "notes": "",
        }
        series_meta[key] = record

    # Enrich records with aggregated b-value data and rule matching
    for key, record in series_meta.items():
        record["total_slices"] = str(series_slice_counts[key])

        b_vals = series_b_values.get(key, [])
        if b_vals:
            unique_sorted = _sort_b_values(b_vals)
            record["unique_b_values"] = ", ".join(unique_sorted)
            counts = Counter(b_vals)
            record["slices_per_b_value"] = ", ".join(
                f"b{v}:{counts[v]}" for v in unique_sorted
            )

        if rules:
            matches = []
            is_t2 = "Rules_T2" in rules and _match_any_selection(
                record, rules["Rules_T2"]
            )
            is_adc = "Rules_ADC" in rules and _match_any_selection(
                record, rules["Rules_ADC"]
            )
            is_highb = "Rules_HIGHB" in rules and _match_any_selection(
                record, rules["Rules_HIGHB"]
            )
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

    records = list(series_meta.values())
    records.sort(
        key=lambda r: (r.get("series_number", ""), r.get("series_description", ""))
    )
    for idx, record in enumerate(records, start=1):
        record["series_group_id"] = str(idx)

    return records


# ---------------------------------------------------------------------------
# Resume / incremental I/O
# ---------------------------------------------------------------------------

def _get_completed_cases(output_path: str) -> Set[str]:
    """Read existing output CSV and return set of case_ids already processed."""
    completed: Set[str] = set()
    if not os.path.exists(output_path):
        return completed
    try:
        with open(output_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get("case_id", "").strip()
                if cid:
                    completed.add(cid)
    except Exception:
        pass
    return completed


def _check_output_columns(output_path: str) -> bool:
    """Return True if the output CSV header matches TEMPLATE_COLUMNS (or doesn't exist)."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return True
    try:
        with open(output_path, "r", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
            return header == TEMPLATE_COLUMNS
    except Exception:
        return False


def _init_output(output_path: str) -> None:
    """Write CSV header if the file doesn't already contain one."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()


def _append_rows(output_path: str, rows: List[Dict[str, str]]) -> None:
    """Thread-safe append of rows to the output CSV."""
    if not rows:
        return
    with _write_lock:
        with open(output_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=TEMPLATE_COLUMNS)
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in TEMPLATE_COLUMNS})


def _discover_cases(test_data_root: str) -> List[str]:
    """Return sorted list of all immediate subdirectory names in *test_data_root*."""
    if not os.path.isdir(test_data_root):
        return []
    return sorted(
        entry
        for entry in os.listdir(test_data_root)
        if os.path.isdir(os.path.join(test_data_root, entry))
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fill_series_description(
    test_data_root: str,
    output_path: str,
    cases: Optional[List[str]] = None,
    app_path: str = "",
    num_workers: int = 8,
) -> str:
    """Scan DICOM directories and write a filled CSV with metadata + b-value analysis.

    Supports resume: if *output_path* already exists with matching columns,
    completed cases are automatically skipped.  Ctrl+C stops gracefully and
    progress is preserved for the next run.

    Parameters
    ----------
    test_data_root : str
        Root directory containing one subfolder per case.
    output_path : str
        Path for the output CSV file.
    cases : list[str] or None
        Case folder names to scan. ``None`` = auto-discover all subdirectories.
    app_path : str
        Path to ``app.py`` containing Rules_T2 / Rules_ADC / Rules_HIGHB.
    num_workers : int
        Number of parallel threads (default 8).
    """
    if cases is None:
        cases = _discover_cases(test_data_root)
        print(f"Auto-discovered {len(cases)} case(s) in {test_data_root}/")

    rules = _parse_rules(app_path) if app_path else {}

    if not _check_output_columns(output_path):
        backup = output_path + ".bak"
        print(f"Existing CSV has incompatible columns — backed up to {backup}")
        os.rename(output_path, backup)

    completed = _get_completed_cases(output_path)
    if completed:
        print(f"Resuming: {len(completed)} case(s) already processed")
    remaining = [c for c in cases if c not in completed]
    print(f"Cases to process: {len(remaining)}")

    if not remaining:
        print("All cases already processed.")
        return output_path

    _init_output(output_path)

    # Graceful shutdown — first Ctrl+C finishes in-progress cases, second forces exit
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig, frame):
        if _shutdown.is_set():
            print("\nForce quit.")
            sys.exit(1)
        print("\n\nInterrupted — finishing in-progress cases, then stopping...")
        print("(Press Ctrl+C again to force quit immediately)\n")
        _shutdown.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    total_series = 0
    processed_count = 0
    start_time = time.time()
    errors: List[Tuple[str, str]] = []

    if not HAS_TQDM:
        print("(tip: pip install tqdm for a progress bar)")

    try:
        pbar = (
            tqdm(
                total=len(remaining),
                desc="Cases",
                unit="case",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
                ),
            )
            if HAS_TQDM
            else None
        )

        def _process_case(
            case_id: str,
        ) -> Tuple[str, List[Dict[str, str]]]:
            return case_id, _extract_series_records(test_data_root, case_id, rules)

        pool = ThreadPoolExecutor(max_workers=num_workers)
        futures = {pool.submit(_process_case, cid): cid for cid in remaining}

        try:
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    _, records = future.result()
                    if records:
                        _append_rows(output_path, records)
                        total_series += len(records)
                        processed_count += 1
                    elif not _shutdown.is_set():
                        processed_count += 1
                except Exception as exc:
                    errors.append((case_id, str(exc)))

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(
                        series=total_series, errors=len(errors), refresh=False
                    )
                else:
                    elapsed = time.time() - start_time
                    pct = (processed_count + len(errors)) / len(remaining) * 100
                    print(
                        f"\r  {processed_count + len(errors)}/{len(remaining)} "
                        f"({pct:.1f}%) | series={total_series} | "
                        f"elapsed={elapsed:.0f}s",
                        end="",
                        flush=True,
                    )

                if _shutdown.is_set():
                    for f in futures:
                        f.cancel()
                    break
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        if pbar:
            pbar.close()

    finally:
        signal.signal(signal.SIGINT, original_sigint)

    elapsed = time.time() - start_time
    hrs, remainder = divmod(int(elapsed), 3600)
    mins, secs = divmod(remainder, 60)
    time_str = f"{hrs}h{mins:02d}m{secs:02d}s" if hrs else f"{mins}m{secs:02d}s"

    print(f"\n{'='*60}")
    print(f"Processed  : {processed_count} case(s), {total_series} series")
    print(f"Elapsed    : {time_str}")
    print(f"Output     : {output_path}")

    if errors:
        print(f"Errors     : {len(errors)}")
        for cid, err in errors[:10]:
            print(f"  {cid}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    still_remaining = len(remaining) - processed_count - len(errors)
    if still_remaining > 0:
        print(f"\nRemaining  : {still_remaining} case(s) — re-run to resume.")
    elif not errors:
        total_done = len(completed) + processed_count
        print(f"\nAll {total_done} case(s) complete.")

    print(f"{'='*60}")
    _shutdown.clear()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fill series description CSV from DICOM headers. "
            "Supports resume (re-run to continue) and multi-threading."
        ),
    )
    parser.add_argument(
        "--test-data-root",
        default=os.path.join(os.getcwd(), "test-data"),
        help="Path to the test-data directory (default: ./test-data).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.getcwd(), "series_description_investigation_filled.csv"),
        help="Path to the output CSV (default: ./series_description_investigation_filled.csv).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Case IDs to scan. If omitted, all subdirectories are scanned.",
    )
    parser.add_argument(
        "--app-path",
        default=os.path.join(os.getcwd(), "prostate_mri_lesion_seg_app", "app.py"),
        help="Path to app.py with selection rules (default: ./prostate_mri_lesion_seg_app/app.py).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel threads (default: 8).",
    )
    args = parser.parse_args()

    fill_series_description(
        test_data_root=args.test_data_root,
        output_path=args.output,
        cases=args.cases,
        app_path=args.app_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
