#!/usr/bin/env python3
"""Validate T2/ADC/HIGHB selection rules against Dr. Anh's annotated CSV.

This harness loads the annotated ground-truth CSV, applies each rule's
regex (with the same semantics as DICOMSeriesSelectorOperator — case-
insensitive re.search), and reports precision / recall / F1 per class
plus false positives and false negatives for inspection.

Usage:
    python scripts/validate_selection_rules.py                          # validate rules in app.py
    python scripts/validate_selection_rules.py --rules-file custom.py   # test custom rules module
    python scripts/validate_selection_rules.py --show-errors            # print FP/FN details
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

DEFAULT_CSV = "series_description_sgh_deid_data.tnt_with_anh_input.csv"
DEFAULT_APP = "prostate_mri_lesion_seg_app/app.py"

GROUND_TRUTH_MAP = {
    "T2W": "T2",
    "ADC": "ADC",
    "DWI": "HIGHB",
    "ignore": None,
}


def _parse_rules(app_path: str) -> Dict[str, Dict[str, object]]:
    """Extract Rules_T2/Rules_ADC/Rules_HIGHB dicts from *app.py*."""
    with open(app_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    out: Dict[str, Dict[str, object]] = {}
    for name, label in (("Rules_T2", "T2"), ("Rules_ADC", "ADC"), ("Rules_HIGHB", "HIGHB")):
        match = re.search(rf"{name}\s*=\s*\"\"\"(.*?)\"\"\"", content, re.DOTALL)
        if not match:
            raise RuntimeError(f"Could not find {name} in {app_path}")
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {name}: {exc}") from exc
        selections = data.get("selections", [])
        if not selections:
            raise RuntimeError(f"{name} has no selections")
        # We validate only on the FIRST selection for simplicity
        conditions = selections[0].get("conditions", {})
        out[label] = conditions
    return out


def _row_matches_conditions(row: Dict[str, str], conditions: Dict[str, object]) -> bool:
    """Mimic DICOMSeriesSelectorOperator._select_series matching for one row.

    Replicates (minus ImageType — see note below):
      - Modality     : exact casefold equality, fallback to regex search
      - SeriesDescription : regex search (case-insensitive)
      - ImageType    : subset test against parsed list (case-insensitive)
    """
    modality_rule = conditions.get("Modality")
    if modality_rule:
        attr = (row.get("modality") or "").strip()
        if attr.casefold() != str(modality_rule).casefold():
            if not re.search(str(modality_rule), attr, re.IGNORECASE):
                return False

    series_desc_rule = conditions.get("SeriesDescription")
    if series_desc_rule:
        desc = row.get("series_description") or ""
        try:
            if not re.search(str(series_desc_rule), desc, re.IGNORECASE):
                return False
        except re.error as exc:
            raise RuntimeError(f"Invalid regex for SeriesDescription: {exc}") from exc

    # Note: ImageType is NOT evaluated here because the annotated CSV
    # doesn't carry it.  We assume Dr. Anh's T2W ground-truth rows all
    # have ImageType=[ORIGINAL, PRIMARY] (which is true by definition —
    # they wouldn't be labelled T2W otherwise).
    return True


def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _predict(row: Dict[str, str], rules: Dict[str, Dict[str, object]]) -> List[str]:
    """Return list of rule labels that match this row (could be 0, 1 or multiple)."""
    hits = []
    for label, conditions in rules.items():
        if _row_matches_conditions(row, conditions):
            hits.append(label)
    return hits


def _evaluate(rows: List[Dict[str, str]], rules: Dict[str, Dict[str, object]]):
    """Compute confusion matrix and collect errors."""
    per_class_stats: Dict[str, Dict[str, int]] = {
        "T2": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "ADC": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "HIGHB": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    multi_match_rows = []
    errors: Dict[str, Dict[str, List[Dict]]] = {
        label: {"fp": [], "fn": []} for label in ("T2", "ADC", "HIGHB")
    }

    for row in rows:
        true_label = GROUND_TRUTH_MAP.get((row.get("manual match") or "").strip())
        hits = _predict(row, rules)

        if len(hits) > 1:
            multi_match_rows.append((row, hits, true_label))

        for label in ("T2", "ADC", "HIGHB"):
            predicted = label in hits
            actual = true_label == label
            stats = per_class_stats[label]
            if predicted and actual:
                stats["tp"] += 1
            elif predicted and not actual:
                stats["fp"] += 1
                errors[label]["fp"].append(row)
            elif not predicted and actual:
                stats["fn"] += 1
                errors[label]["fn"].append(row)
            else:
                stats["tn"] += 1

    return per_class_stats, errors, multi_match_rows


def _print_report(per_class_stats, errors, multi_match_rows, rows, show_errors: bool):
    print("\n" + "=" * 78)
    print("  SELECTION RULES VALIDATION REPORT")
    print("=" * 78)
    print(f"  Rows evaluated : {len(rows):,}")
    gt_counts = Counter(
        GROUND_TRUTH_MAP.get((r.get("manual match") or "").strip(), "ignore")
        for r in rows
    )
    print("  Ground truth distribution:")
    for label in ("T2", "ADC", "HIGHB", None):
        display = label if label else "ignore"
        print(f"     {display:<7}: {gt_counts.get(label, 0):>6}")
    print()

    def fmt(n: float) -> str:
        return f"{n*100:6.2f}%" if n == n else "   n/a "

    print(f"  {'Class':<8} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10}")
    print("  " + "-" * 66)
    for label in ("T2", "ADC", "HIGHB"):
        s = per_class_stats[label]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else float("nan")
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else float("nan")
        f1 = 2 * p * r / (p + r) if p and r and p + r > 0 else float("nan")
        print(f"  {label:<8} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6} "
              f"{fmt(p):>10} {fmt(r):>10} {fmt(f1):>10}")
    print()

    if multi_match_rows:
        print(f"  {len(multi_match_rows)} row(s) matched MULTIPLE rules (ambiguity).")
        sample = Counter(
            (tuple(sorted(hits)), row.get("series_description", ""))
            for row, hits, _ in multi_match_rows[:5000]
        )
        print("  Sample multi-match descriptions:")
        for (labels, desc), n in sample.most_common(10):
            print(f"    {n:>5}  {'+'.join(labels):<12} {desc!r}")
        print()

    for label in ("T2", "ADC", "HIGHB"):
        s = per_class_stats[label]
        total_errors = s["fp"] + s["fn"]
        if total_errors == 0:
            continue
        print(f"  === {label} errors ===")
        fp_descs = Counter(r.get("series_description", "") for r in errors[label]["fp"])
        fn_descs = Counter(r.get("series_description", "") for r in errors[label]["fn"])
        if fp_descs:
            print(f"  FALSE POSITIVES (predicted {label} but should be other): "
                  f"{s['fp']} rows / {len(fp_descs)} unique descriptions")
            for desc, n in fp_descs.most_common(15 if not show_errors else 999):
                print(f"    {n:>5}  {desc!r}")
            if not show_errors and len(fp_descs) > 15:
                print(f"    ... {len(fp_descs) - 15} more unique (use --show-errors to see all)")
        if fn_descs:
            print(f"  FALSE NEGATIVES (should have matched {label}): "
                  f"{s['fn']} rows / {len(fn_descs)} unique descriptions")
            for desc, n in fn_descs.most_common(15 if not show_errors else 999):
                print(f"    {n:>5}  {desc!r}")
            if not show_errors and len(fn_descs) > 15:
                print(f"    ... {len(fn_descs) - 15} more unique (use --show-errors to see all)")
        print()
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"Annotated CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--app-path", default=DEFAULT_APP,
                        help=f"app.py with rules (default: {DEFAULT_APP})")
    parser.add_argument("--show-errors", action="store_true",
                        help="Print full FP/FN lists instead of top-15")
    parser.add_argument("--override-rules",
                        help="JSON file with custom rules, format: "
                             '{"T2":{"SeriesDescription":"..."},"ADC":{...},"HIGHB":{...}}')
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    if args.override_rules:
        with open(args.override_rules, "r", encoding="utf-8") as fh:
            rules = json.load(fh)
        print(f"Loaded rules from override file: {args.override_rules}")
    else:
        app_path = args.app_path
        if not os.path.exists(app_path):
            print(f"ERROR: app.py not found: {app_path}", file=sys.stderr)
            return 1
        rules = _parse_rules(app_path)
        print(f"Loaded rules from: {app_path}")

    rows = _load_rows(csv_path)
    per_class_stats, errors, multi_match_rows = _evaluate(rows, rules)
    _print_report(per_class_stats, errors, multi_match_rows, rows, args.show_errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
