from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    Spacingd,
    SpatialPadd,
)
from skimage.transform import resize


def get_organ_train_transforms(
    spacing: Sequence[float],
    roi_size: Sequence[int],
    samples_per_image: int,
    pos_to_neg: float,
) -> Compose:
    """T2-only binary whole-gland transforms matching organ inference."""
    if pos_to_neg <= 0:
        raise ValueError("preprocess.pos_to_neg_ratio must be positive")
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(spacing),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(
                keys=["image"], nonzero=True, channel_wise=True
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(roi_size)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(roi_size),
                pos=float(pos_to_neg),
                neg=1.0,
                num_samples=int(samples_per_image),
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.05, 0.05, 0.03),
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.15, std=0.05),
            EnsureTyped(keys=["image", "label"], dtype=(np.float32, np.int64)),
        ]
    )


def get_organ_val_transforms(spacing: Sequence[float]) -> Compose:
    """Deterministic organ validation transforms matching the deployed app."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(spacing),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(
                keys=["image"], nonzero=True, channel_wise=True
            ),
            EnsureTyped(keys=["image", "label"], dtype=(np.float32, np.int64)),
        ]
    )


def _ensure_sitk_3d(image: sitk.Image) -> sitk.Image:
    return sitk.JoinSeries([image]) if image.GetDimension() == 2 else image


def _ensure_numpy_3d(array: np.ndarray) -> np.ndarray:
    return array[..., np.newaxis] if array.ndim == 2 else array


def _canonical_data(path: str | Path) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    return _ensure_numpy_3d(image.get_fdata())


def _align_modality_to_t2(modality_path: str, t2_path: str, output_path: Path) -> None:
    """Mirror the deployed lesion operator's SimpleITK nearest-neighbour alignment."""
    reference = _ensure_sitk_3d(sitk.ReadImage(t2_path))
    moving = _ensure_sitk_3d(sitk.ReadImage(modality_path))
    aligned = sitk.Resample(
        moving,
        reference.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        reference.GetOrigin(),
        reference.GetSpacing(),
        reference.GetDirection(),
        0,
        reference.GetPixelID(),
    )
    sitk.WriteImage(aligned, str(output_path))


def normalize_lesion_channels(image: np.ndarray) -> np.ndarray:
    """Per-channel full-ROI z-score used by lesion inference."""
    output = np.asarray(image, dtype=np.float32).copy()
    for channel in range(output.shape[0]):
        values = output[channel]
        if float(np.max(np.abs(values))) < 1e-7:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values))
        output[channel] = values - mean if std < 1e-7 else (values - mean) / std
    return output


def bbox_with_inference_margin(mask: np.ndarray, margin: int) -> tuple[slice, ...]:
    """Return the app's intentionally exclusive high-bound ROI slices."""
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        raise ValueError("cleaned organ mask is empty")
    low = foreground.min(axis=0)
    high = foreground.max(axis=0)
    shape = np.asarray(mask.shape)
    low = np.maximum(0, low - int(margin))
    high = np.minimum(shape - 1, high + int(margin))
    slices = tuple(slice(int(low[i]), int(high[i])) for i in range(3))
    if any(item.stop <= item.start for item in slices):
        raise ValueError("organ ROI is empty after cropping")
    return slices


def preprocess_lesion_case(
    record: Mapping[str, str],
    *,
    spacing: Sequence[float] = (0.5, 0.5, 0.5),
    margin: int = 32,
) -> dict[str, np.ndarray]:
    """Build the exact T2/ADC/HIGHB lesion input and aligned binary targets.

    Source files are never modified. ADC and HIGHB are aligned in a temporary
    directory because the deployed app currently materialises that alignment
    before nibabel canonicalisation.
    """
    t2_path = record["t2_path"]
    t2_image = nib.load(t2_path)
    original_spacing = tuple(t2_image.header.get_zooms())
    if len(original_spacing) < 3:
        original_spacing += (1.0,) * (3 - len(original_spacing))

    t2 = _canonical_data(t2_path)
    with tempfile.TemporaryDirectory(prefix="lesion-align-") as temp_dir:
        temp_root = Path(temp_dir)
        modalities = [t2]
        for key in ("adc_path", "highb_path"):
            output = temp_root / f"{key}.nii.gz"
            _align_modality_to_t2(record[key], t2_path, output)
            modalities.append(_canonical_data(output))

    image = np.stack(modalities, axis=0).astype(np.float32)
    source_shape = image.shape[1:]
    organ = (_canonical_data(record["organ_label_path"]) > 0).astype(np.uint8)
    label = (_canonical_data(record["lesion_label_path"]) > 0).astype(np.uint8)
    if organ.shape != source_shape:
        organ = (resize(organ, source_shape, order=0, preserve_range=True) > 0).astype(
            np.uint8
        )
    if label.shape != source_shape:
        label = (resize(label, source_shape, order=0, preserve_range=True) > 0).astype(
            np.uint8
        )

    target_shape = tuple(
        max(1, int(round(source_shape[i] * original_spacing[i] / float(spacing[i]))))
        for i in range(3)
    )
    resized_image = np.stack(
        [
            resize(channel, target_shape, order=1, preserve_range=True).astype(
                np.float32
            )
            for channel in image
        ],
        axis=0,
    )
    resized_organ = (
        resize(organ, target_shape, order=0, preserve_range=True) > 0
    ).astype(np.uint8)
    resized_label = (
        resize(label, target_shape, order=0, preserve_range=True) > 0
    ).astype(np.uint8)

    roi = bbox_with_inference_margin(resized_organ, margin)
    cropped_image = normalize_lesion_channels(resized_image[(slice(None),) + roi])
    cropped_organ = resized_organ[roi]
    cropped_label = resized_label[roi]
    return {
        "image": cropped_image.astype(np.float32),
        "label": cropped_label.astype(np.uint8),
        "organ": cropped_organ.astype(np.uint8),
        "source_shape": np.asarray(source_shape, dtype=np.int32),
        "resampled_shape": np.asarray(target_shape, dtype=np.int32),
        "bbox": np.asarray(
            [value for item in roi for value in (item.start, item.stop)],
            dtype=np.int32,
        ),
    }
