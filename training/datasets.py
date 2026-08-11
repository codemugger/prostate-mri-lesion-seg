from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from monai.data import CacheDataset, Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

from .cohort import read_manifest
from .transforms import preprocess_lesion_case


LESION_CACHE_VERSION = "v2-float32-inference-matched"


def _is_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_split_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "subject_id", "split"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Split CSV is missing column(s): {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    identities = [(row["source"], row["subject_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Split CSV contains duplicate source/subject_id pairs")
    return rows


def select_records(
    manifest_path: str | Path,
    split_path: str | Path,
    *,
    task: str,
    partition: str,
    fold: int | None = None,
) -> list[dict[str, str]]:
    """Join a manifest to a locked split without allowing case leakage."""
    if task not in {"organ", "lesion"}:
        raise ValueError("task must be organ or lesion")
    if partition not in {"train", "val", "test"}:
        raise ValueError("partition must be train, val, or test")
    if task == "lesion" and partition != "test" and fold not in range(5):
        raise ValueError("lesion train/val selection requires fold 0..4")

    manifest = {
        (row["source"], row["subject_id"]): row
        for row in read_manifest(Path(manifest_path))
        if _is_true(row[f"{task}_eligible"])
    }
    selected: list[dict[str, str]] = []
    for assignment in read_split_rows(split_path):
        identity = (assignment["source"], assignment["subject_id"])
        record = manifest.get(identity)
        if record is None:
            continue
        if task == "organ":
            include = assignment["split"] == partition
        elif partition == "test":
            include = assignment["split"] == "test"
        elif partition == "val":
            include = (
                assignment["split"] == "development"
                and assignment.get("fold", "") == str(fold)
            )
        else:
            include = (
                assignment["split"] == "development"
                and assignment.get("fold", "") != str(fold)
            )
        if include:
            selected.append(dict(record))
    selected.sort(key=lambda row: (row["source"], row["subject_id"]))
    if not selected:
        raise ValueError(f"No {task} records selected for partition={partition}")
    return selected


def build_organ_datasets(
    train_records: Sequence[Mapping[str, str]],
    val_records: Sequence[Mapping[str, str]],
    *,
    train_transform,
    val_transform,
    cache_rate: float,
    num_workers: int,
) -> tuple[CacheDataset, Dataset]:
    def items(records: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        return [
            {
                "image": row["t2_path"],
                "label": row["organ_label_path"],
                "subject_id": row["subject_id"],
                "source": row["source"],
            }
            for row in records
        ]

    train = CacheDataset(
        items(train_records),
        transform=train_transform,
        cache_rate=float(cache_rate),
        num_workers=int(num_workers),
    )
    validation = Dataset(items(val_records), transform=val_transform)
    return train, validation


def source_balanced_sampler(
    records: Sequence[Mapping[str, str]],
    repeats: int = 1,
    source_weights: Mapping[str, float] | None = None,
):
    """Create an explicit source-mixture sampler.

    If weights are omitted each source receives equal mass. Callers should use
    ordinary shuffled sampling when they want the natural cohort proportions.
    """
    sources = [row["source"] for row in records for _ in range(int(repeats))]
    counts = Counter(sources)
    target = (
        {source: 1.0 for source in counts}
        if source_weights is None
        else {str(key): float(value) for key, value in source_weights.items()}
    )
    missing = sorted(set(counts) - set(target))
    if missing:
        raise ValueError(
            "dataset.source_sampling_weights is missing source(s): "
            + ", ".join(missing)
        )
    if any(target[source] <= 0 for source in counts):
        raise ValueError("All source sampling weights must be positive")
    weights = torch.as_tensor(
        [target[source] / counts[source] for source in sources], dtype=torch.double
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _cache_key(record: Mapping[str, str], spacing: Sequence[float], margin: int) -> str:
    values = [LESION_CACHE_VERSION, str(tuple(spacing)), str(margin)]
    for key in (
        "t2_path",
        "adc_path",
        "highb_path",
        "organ_label_path",
        "lesion_label_path",
    ):
        path = Path(record[key])
        stat = path.stat()
        values.extend([str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)])
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


class LesionDataset(TorchDataset):
    """On-demand inference-matched lesion preprocessing with a persistent cache."""

    def __init__(
        self,
        records: Sequence[Mapping[str, str]],
        *,
        cache_dir: str | Path,
        spacing: Sequence[float],
        margin: int,
        patch_size: Sequence[int] | None,
        samples_per_case: int = 1,
        positive_fraction: float = 0.67,
        augment: bool = False,
    ):
        self.records = [dict(record) for record in records]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.spacing = tuple(float(value) for value in spacing)
        self.margin = int(margin)
        self.patch_size = (
            tuple(int(value) for value in patch_size) if patch_size is not None else None
        )
        self.samples_per_case = int(samples_per_case)
        self.positive_fraction = float(positive_fraction)
        self.augment = bool(augment)
        if self.samples_per_case < 1:
            raise ValueError("samples_per_case must be >= 1")
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be between 0 and 1")

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_case

    def _load(self, record: Mapping[str, str]) -> dict[str, np.ndarray]:
        cache_path = self.cache_dir / (
            _cache_key(record, self.spacing, self.margin) + ".npz"
        )
        if not cache_path.is_file():
            prepared = preprocess_lesion_case(
                record, spacing=self.spacing, margin=self.margin
            )
            temporary = cache_path.with_name(
                f".{cache_path.stem}.{os.getpid()}.tmp.npz"
            )
            np.savez_compressed(
                temporary,
                image=prepared["image"].astype(np.float32),
                label=prepared["label"].astype(np.uint8),
                organ=prepared["organ"].astype(np.uint8),
            )
            try:
                os.replace(temporary, cache_path)
            except FileNotFoundError:
                pass
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                return {
                    "image": data["image"].astype(np.float32),
                    "label": data["label"].astype(np.uint8),
                    "organ": data["organ"].astype(np.uint8),
                }
        except (OSError, ValueError):
            cache_path.unlink(missing_ok=True)
            return self._load(record)

    @staticmethod
    def _pad(
        image: np.ndarray, label: np.ndarray, minimum: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        padding = []
        for size, target in zip(image.shape[1:], minimum):
            total = max(0, int(target) - int(size))
            padding.append((total // 2, total - total // 2))
        if any(before or after for before, after in padding):
            image = np.pad(image, [(0, 0), *padding], mode="constant")
            label = np.pad(label, padding, mode="constant")
        return image, label

    def _crop_patch(
        self, image: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.patch_size is not None
        image, label = self._pad(image, label, self.patch_size)
        spatial = np.asarray(image.shape[1:])
        patch = np.asarray(self.patch_size)
        positives = np.argwhere(label > 0)
        use_positive = positives.size and np.random.random() < self.positive_fraction
        if use_positive:
            center = positives[np.random.randint(len(positives))]
        else:
            center = np.asarray(
                [np.random.randint(int(size)) for size in spatial], dtype=np.int64
            )
        start = np.minimum(np.maximum(center - patch // 2, 0), spatial - patch)
        slices = tuple(
            slice(int(start[i]), int(start[i] + patch[i])) for i in range(3)
        )
        return image[(slice(None),) + slices], label[slices]

    @staticmethod
    def _pad_for_network(
        image: np.ndarray, label: np.ndarray, multiple: int = 32
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
        original = tuple(int(value) for value in label.shape)
        target = tuple(max(multiple, size) for size in original)
        image, label = LesionDataset._pad(image, label, target)
        return image, label, original

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index // self.samples_per_case]
        prepared = self._load(record)
        image, label, organ = (
            prepared["image"],
            prepared["label"],
            prepared["organ"],
        )
        original_shape = tuple(int(value) for value in label.shape)
        if self.patch_size is not None:
            image, label = self._crop_patch(image, label)
        else:
            image, label, original_shape = self._pad_for_network(image, label)
            target_shape = label.shape
            padding = []
            for size, target in zip(organ.shape, target_shape):
                total = max(0, int(target) - int(size))
                padding.append((total // 2, total - total // 2))
            organ = np.pad(organ, padding, mode="constant")

        if self.augment:
            for axis in range(3):
                if np.random.random() < 0.5:
                    image = np.flip(image, axis=axis + 1)
                    label = np.flip(label, axis=axis)
            if np.random.random() < 0.15:
                image = image + np.random.normal(0.0, 0.05, image.shape).astype(
                    np.float32
                )

        result: dict[str, object] = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "label": torch.from_numpy(
                np.ascontiguousarray(label[np.newaxis])
            ).long(),
            "subject_id": record["subject_id"],
            "source": record["source"],
            "original_shape": torch.as_tensor(original_shape, dtype=torch.int64),
        }
        if self.patch_size is None:
            result["organ"] = torch.from_numpy(
                np.ascontiguousarray(organ[np.newaxis])
            ).to(torch.uint8)
        return result
