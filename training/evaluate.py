from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from monai.data import Dataset
from monai.inferers import SlidingWindowInferer

from .cohort import clean_binary_organ_mask, write_csv
from .datasets import LesionDataset, select_records
from .engine import build_model
from .export import load_weights, sha256_file
from .inference_tiler import lesion_tiled_predict
from .metrics import binary_case_metrics, lesion_detection_counts
from .transforms import get_organ_val_transforms
from .utils import load_yaml


RESULT_COLUMNS = [
    "source",
    "subject_id",
    "manufacturer",
    "model",
    "study_date",
    "study_year",
    "dice",
    "precision",
    "recall",
    "detected_lesions",
    "missed_lesions",
    "false_positive_lesions",
]


def _record_result(
    record: Mapping[str, str],
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    task: str,
) -> dict[str, object]:
    metrics = binary_case_metrics(prediction, target)
    result: dict[str, object] = {
        key: record.get(key, "")
        for key in (
            "source",
            "subject_id",
            "manufacturer",
            "model",
            "study_date",
            "study_year",
        )
    }
    result.update(
        {key: round(float(metrics[key]), 6) for key in ("dice", "precision", "recall")}
    )
    if task == "lesion":
        detected, missed, false_positive = lesion_detection_counts(
            prediction, target
        )
        result.update(
            {
                "detected_lesions": detected,
                "missed_lesions": missed,
                "false_positive_lesions": false_positive,
            }
        )
    else:
        result.update(
            {
                "detected_lesions": "",
                "missed_lesions": "",
                "false_positive_lesions": "",
            }
        )
    return result


def _evaluate_organ(
    cfg: Mapping,
    records: Sequence[Mapping[str, str]],
    model_path: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    model = build_model(cfg).to(device).eval()
    load_weights(model, model_path, map_location=device)
    transform = get_organ_val_transforms(cfg["preprocess"]["spacing"])
    dataset = Dataset(
        [
            {
                "image": record["t2_path"],
                "label": record["organ_label_path"],
            }
            for record in records
        ],
        transform=transform,
    )
    validation_cfg = cfg["validation"]
    inferer = SlidingWindowInferer(
        roi_size=tuple(validation_cfg["roi_size"]),
        sw_batch_size=int(validation_cfg["sw_batch_size"]),
        overlap=float(validation_cfg["overlap"]),
        mode="gaussian",
    )
    results = []
    with torch.inference_mode():
        for index, record in enumerate(records):
            sample = dataset[index]
            logits = inferer(sample["image"].unsqueeze(0).to(device), model)
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()
            target = sample["label"][0].cpu().numpy() > 0
            cleaned, _ = clean_binary_organ_mask(prediction)
            results.append(_record_result(record, cleaned, target, task="organ"))
            print(f"  organ test {index + 1}/{len(records)}", flush=True)
    return results


def _evaluate_lesion(
    cfg: Mapping,
    records: Sequence[Mapping[str, str]],
    model_paths: Sequence[Path],
    device: torch.device,
) -> list[dict[str, object]]:
    if len(model_paths) != 5:
        raise ValueError("Lesion held-out evaluation requires five fold checkpoints")
    models = []
    for fold, path in enumerate(model_paths):
        model = build_model(cfg).to(device).eval()
        metadata = load_weights(model, path, map_location=device)
        if metadata.get("fold") not in {None, fold}:
            raise ValueError(
                f"Checkpoint {path} declares fold={metadata.get('fold')}, "
                f"expected fold={fold}"
            )
        models.append(model)
    preprocessing = cfg["preprocess"]
    dataset = LesionDataset(
        records,
        cache_dir=cfg["dataset"]["cache_dir"],
        spacing=preprocessing["spacing"],
        margin=int(preprocessing["roi_margin"]),
        patch_size=None,
        samples_per_case=1,
        augment=False,
    )
    threshold = float(cfg["validation"].get("ensemble_threshold", 0.63))
    results = []
    with torch.inference_mode():
        for index, record in enumerate(records):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0).to(device)
            probability = torch.stack(
                [lesion_tiled_predict(model, image) for model in models], dim=0
            ).mean(dim=0)
            organ = sample["organ"][0].cpu().numpy() > 0
            prediction = (
                probability[0, 1].cpu().numpy() * organ
            ) >= threshold
            target = sample["label"][0].cpu().numpy() > 0
            results.append(
                _record_result(record, prediction, target, task="lesion")
            )
            print(f"  lesion test {index + 1}/{len(records)}", flush=True)
    return results


def _write_results(
    output_dir: Path, results: Sequence[Mapping[str, object]], *, task: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_case_metrics.csv", results, RESULT_COLUMNS)
    summary = {
        "task": task,
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "cases": len(results),
        "mean_dice": float(np.mean([float(row["dice"]) for row in results])),
        "mean_precision": float(
            np.mean([float(row["precision"]) for row in results])
        ),
        "mean_recall": float(np.mean([float(row["recall"]) for row in results])),
    }
    if task == "lesion":
        detected = sum(int(row["detected_lesions"]) for row in results)
        missed = sum(int(row["missed_lesions"]) for row in results)
        false_positive = sum(
            int(row["false_positive_lesions"]) for row in results
        )
        summary.update(
            {
                "lesion_sensitivity": detected / max(detected + missed, 1),
                "false_positive_lesions_per_case": false_positive
                / max(len(results), 1),
                "detected_lesions": detected,
                "missed_lesions": missed,
            }
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    group_columns = [
        "group_field",
        "group_value",
        "cases",
        "mean_dice",
        "mean_precision",
        "mean_recall",
    ]
    for field in ("manufacturer", "model", "study_year"):
        groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in results:
            groups[str(row.get(field, "") or "UNKNOWN")].append(row)
        rows = []
        for value, group in sorted(groups.items()):
            rows.append(
                {
                    "group_field": field,
                    "group_value": value,
                    "cases": len(group),
                    "mean_dice": round(
                        float(np.mean([float(row["dice"]) for row in group])), 6
                    ),
                    "mean_precision": round(
                        float(np.mean([float(row["precision"]) for row in group])),
                        6,
                    ),
                    "mean_recall": round(
                        float(np.mean([float(row["recall"]) for row in group])), 6
                    ),
                }
            )
        write_csv(output_dir / f"metrics_by_{field}.csv", rows, group_columns)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate only the locked held-out test cohort."
    )
    parser.add_argument("--task", choices=("organ", "lesion"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        help="One organ model or five lesion fold models in fold order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    records = select_records(
        cfg["dataset"]["manifest"],
        cfg["dataset"]["split_csv"],
        task=args.task,
        partition="test",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "organ":
        if len(args.model) != 1:
            raise ValueError("Organ evaluation requires exactly one --model")
        results = _evaluate_organ(cfg, records, args.model[0], device)
    else:
        results = _evaluate_lesion(cfg, records, args.model, device)
    _write_results(args.output_dir, results, task=args.task)
    provenance = {
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config)),
        "manifest": str(Path(cfg["dataset"]["manifest"]).resolve()),
        "manifest_sha256": sha256_file(Path(cfg["dataset"]["manifest"])),
        "split_csv": str(Path(cfg["dataset"]["split_csv"]).resolve()),
        "split_sha256": sha256_file(Path(cfg["dataset"]["split_csv"])),
        "models": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in args.model
        ],
    }
    (args.output_dir / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
