from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label as connected_components


def binary_case_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    if not prediction.any() and not target.any():
        dice = 1.0
    else:
        dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    return {
        "dice": float(dice),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def lesion_detection_counts(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[int, int, int]:
    """Count detected GT lesions, missed GT lesions, and false-positive components."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    predicted_components, predicted_count = connected_components(
        np.asarray(prediction, dtype=bool), structure=structure
    )
    target_components, target_count = connected_components(
        np.asarray(target, dtype=bool), structure=structure
    )
    detected = sum(
        bool(np.any(prediction[target_components == component]))
        for component in range(1, target_count + 1)
    )
    false_positive = sum(
        not bool(np.any(target[predicted_components == component]))
        for component in range(1, predicted_count + 1)
    )
    return detected, target_count - detected, false_positive


@dataclass
class SegmentationMetrics:
    task: str
    cases: list[dict[str, float]] = field(default_factory=list)
    detected_lesions: int = 0
    missed_lesions: int = 0
    false_positive_lesions: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        self.cases.append(binary_case_metrics(prediction, target))
        if self.task == "lesion":
            detected, missed, false_positive = lesion_detection_counts(
                prediction, target
            )
            self.detected_lesions += detected
            self.missed_lesions += missed
            self.false_positive_lesions += false_positive

    def compute(self) -> dict[str, float]:
        if not self.cases:
            return {"dice": 0.0, "precision": 0.0, "recall": 0.0}
        result = {
            key: float(np.mean([case[key] for case in self.cases]))
            for key in ("dice", "precision", "recall")
        }
        if self.task == "lesion":
            total_lesions = self.detected_lesions + self.missed_lesions
            result.update(
                {
                    "lesion_sensitivity": float(
                        self.detected_lesions / max(total_lesions, 1)
                    ),
                    "false_positive_lesions_per_case": float(
                        self.false_positive_lesions / len(self.cases)
                    ),
                    "detected_lesions": float(self.detected_lesions),
                    "missed_lesions": float(self.missed_lesions),
                }
            )
        return result
