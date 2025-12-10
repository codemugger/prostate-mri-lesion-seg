from __future__ import annotations

from typing import Sequence, Dict, Any
import numpy as np
import nibabel as nib
from skimage.transform import resize

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    EnsureTyped,
    AsDiscreted,
    RandSpatialCropSamplesd,
    Resized,
    MapTransform,
)
from monai.config import KeysCollection
from monai.data import MetaTensor
import torch


class CreateEmptyLabelIfMissing(MapTransform):
    """
    Creates an empty label (all zeros) if the label key is None or missing.
    This handles cases where no lesion mask exists (no tumor detected).
    Works with images loaded by LoadImaged (can be numpy arrays or MetaTensors).
    """
    def __init__(self, keys: KeysCollection, image_key: str = "image"):
        super().__init__(keys)
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)
        if self.image_key not in d:
            return d  # Can't create label without reference image
        
        image = d[self.image_key]
        
        # Get spatial shape from image (handles various formats from LoadImaged)
        import numpy as np
        
        if isinstance(image, MetaTensor):
            # MetaTensor: shape could be [H, W, D] or [1, H, W, D] or [C, H, W, D]
            img_shape = tuple(image.shape)
            if len(img_shape) == 3:
                image_spatial_shape = img_shape  # [H, W, D]
            elif len(img_shape) == 4:
                image_spatial_shape = img_shape[1:]  # [C, H, W, D] -> [H, W, D]
            else:
                image_spatial_shape = img_shape[-3:]  # Take last 3 dims
        elif isinstance(image, np.ndarray):
            # Numpy array: shape could be [H, W, D] or [1, H, W, D] or [C, H, W, D]
            img_shape = image.shape
            if len(img_shape) == 3:
                image_spatial_shape = img_shape  # [H, W, D]
            elif len(img_shape) == 4:
                image_spatial_shape = img_shape[1:]  # [C, H, W, D] -> [H, W, D]
            else:
                image_spatial_shape = img_shape[-3:]  # Take last 3 dims
        elif hasattr(image, 'shape'):
            # Generic object with shape attribute
            img_shape = tuple(image.shape)
            if len(img_shape) == 3:
                image_spatial_shape = img_shape
            elif len(img_shape) == 4:
                image_spatial_shape = img_shape[1:]
            else:
                image_spatial_shape = img_shape[-3:]
        else:
            # Fallback: try to convert
            img_array = np.asarray(image)
            if len(img_array.shape) == 3:
                image_spatial_shape = img_array.shape
            else:
                image_spatial_shape = img_array.shape[-3:]
        
        for key in self.keys:
            if key not in d or d[key] is None:
                # Create empty label (all zeros) matching image spatial shape
                empty_label = np.zeros(image_spatial_shape, dtype=np.uint8)
                # Preserve metadata from image if available
                if isinstance(image, MetaTensor):
                    d[key] = MetaTensor(empty_label, meta=image.meta)
                else:
                    d[key] = empty_label
        
        return d


class DropNoneKeys(MapTransform):
    """
    Drops specified keys if their value is None or an empty string.
    Use before LoadImaged to avoid None filenames.
    """
    def __call__(self, data):
        d = dict(data)
        for key in list(self.keys):
            if key in d and (d[key] is None or (isinstance(d[key], str) and not d[key].strip())):
                d.pop(key, None)
        return d


class ResizeLabelToMatchImage(MapTransform):
    """
    Custom transform to resize label to match image size after resampling.
    This handles cases where Spacingd results in different sizes for image and label.
    """
    def __init__(self, keys: KeysCollection, image_key: str = "image", mode: str = "nearest"):
        super().__init__(keys)
        self.image_key = image_key
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        image = d[self.image_key]
        
        # Get spatial shape (exclude channel dimension)
        # Convert to tensor if needed to get shape
        if isinstance(image, MetaTensor):
            image_spatial_shape = tuple(image.shape[1:])  # Exclude channel dimension
        elif hasattr(image, 'shape'):
            image_spatial_shape = tuple(image.shape[1:])  # Exclude channel dimension
        else:
            # Try to convert to array/tensor
            import numpy as np
            if isinstance(image, np.ndarray):
                image_spatial_shape = tuple(image.shape[1:])
            else:
                image_spatial_shape = tuple(image.shape[1:])
        
        for key in self.keys:
            if key not in d or d[key] is None:
                continue  # Skip if key is missing or None
                
            label = d[key]
            
            # Get spatial shape (exclude channel dimension)
            if isinstance(label, MetaTensor):
                label_spatial_shape = tuple(label.shape[1:])  # Exclude channel dimension
            elif hasattr(label, 'shape'):
                label_spatial_shape = tuple(label.shape[1:])  # Exclude channel dimension
            else:
                import numpy as np
                if isinstance(label, np.ndarray):
                    label_spatial_shape = tuple(label.shape[1:])
                else:
                    label_spatial_shape = tuple(label.shape[1:])
            
            # Always resize to ensure exact match (handles floating point rounding issues)
            if label_spatial_shape != image_spatial_shape:
                # Resize label to match image size
                resized_transform = Resized(
                    keys=[key],
                    spatial_size=image_spatial_shape,
                    mode=self.mode
                )
                resized_data = resized_transform({key: label})
                d[key] = resized_data[key]
        
        return d


class LabelToOneHot:
    """
    Convert integer labels to one-hot format while preserving batch and channel semantics.
    
    Expected inputs:
      - [B, 1, H, W, D] integer labels (common after EnsureChannelFirstd)
      - [B, H, W, D] integer labels
      - [1, H, W, D] or [H, W, D] also supported (will be treated as batch size 1)
    
    Output:
      - [B, C, H, W, D] float tensor with one-hot channels, where C == num_classes
    
    Notes:
      - Uses scatter_ along channel dimension (dim=1)
      - Returns float32 one-hot to match typical metric expectations and prediction dtype
    """
    def __init__(self, num_classes: int):
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = num_classes

    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(labels):
            raise TypeError("LabelToOneHot expects a torch.Tensor")
        # Normalize shapes to [B, 1, H, W, D] or [B, H, W, D]
        if labels.dim() == 3:
            # [H, W, D] -> [1, H, W, D]
            labels = labels.unsqueeze(0)
        if labels.dim() == 4:
            # [B, H, W, D] -> add channel dim: [B, 1, H, W, D]
            labels = labels.unsqueeze(1)
        if labels.dim() != 5:
            raise ValueError(f"LabelToOneHot expects 3D or 4D volumes with optional batch/channel. Got shape {tuple(labels.shape)}")

        if labels.shape[1] == 0:
            raise ValueError("Channel dimension cannot be zero for labels")

        # Squeeze single channel if present
        if labels.shape[1] == 1:
            label_indices = labels[:, 0, ...]
        else:
            # If multiple channels provided, assume they are already class indices per voxel via argmax
            # Convert to indices along channel dimension
            label_indices = labels.argmax(dim=1)

        label_indices = label_indices.long().clamp(min=0, max=self.num_classes - 1)
        batch_size, height, width, depth = label_indices.shape
        one_hot = torch.zeros(
            (batch_size, self.num_classes, height, width, depth),
            device=labels.device,
            dtype=torch.float32,
        )
        one_hot.scatter_(1, label_indices.unsqueeze(1), 1.0)
        return one_hot


class PredToOneHot:
    """
    Convert raw network logits to discrete one-hot predictions with [B, C, H, W, D].
    
    Accepts:
      - [B, C, H, W, D] logits
      - [C, H, W, D] logits (batch size 1)
    
    Returns:
      - [B, C, H, W, D] one-hot tensor (float32)
    """
    def __init__(self, num_classes: int):
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(logits):
            raise TypeError("PredToOneHot expects a torch.Tensor")
        if logits.dim() == 4:
            # [C, H, W, D] -> add batch dim
            logits = logits.unsqueeze(0)
        if logits.dim() != 5:
            raise ValueError(f"PredToOneHot expects logits with shape [B,C,H,W,D] or [C,H,W,D], got {tuple(logits.shape)}")
        if logits.shape[1] != self.num_classes:
            # Accept mismatch but proceed; argmax doesn't depend on configured num_classes strictly.
            pass
        argmax = torch.argmax(logits, dim=1)  # [B, H, W, D]
        batch_size, height, width, depth = argmax.shape
        one_hot = torch.zeros(
            (batch_size, self.num_classes, height, width, depth),
            device=logits.device,
            dtype=torch.float32,
        )
        one_hot.scatter_(1, argmax.unsqueeze(1), 1.0)
        return one_hot

def get_organ_train_transforms(spacing: Sequence[float], roi_size: Sequence[int], samples_per_image: int, pos_to_neg: float):
    """
    Training transforms for multi-class organ segmentation (background=0, TZ=1, PZ=2).
    - Standardize geometry to RAS and given spacing (1.0mm).
    - Ensure image and label have matching sizes after resampling.
    - Random spatial cropping (works with multi-class labels, unlike RandCropByPosNegLabeld).
    - Light augmentations (flip, small affine, noise).
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            # Ensure label matches image size after resampling (handles size mismatches)
            ResizeLabelToMatchImage(keys=["label"], image_key="image", mode="nearest"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Use RandSpatialCropSamplesd instead of RandCropByPosNegLabeld for multi-class labels
            # This avoids the bitwise operation issue with multi-class labels
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=roi_size,
                num_samples=samples_per_image,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.15,
                rotate_range=(0.0, 0.0, 0.261799),  # ~15 degrees around z as typical for axial MRI
                scale_range=(0.1, 0.1, 0.0),
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_organ_val_transforms(spacing: Sequence[float]):
    """
    Validation transforms for multi-class organ segmentation (background=0, TZ=1, PZ=2).
    - Ensure image and label have matching sizes after resampling.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            # Ensure label matches image size after resampling (handles size mismatches)
            ResizeLabelToMatchImage(keys=["label"], image_key="image", mode="nearest"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


class StackMultiModalImages(MapTransform):
    """
    Stacks T2, ADC, and HIGHB into a 3-channel image.
    Handles missing ADC/HIGHB by using zeros matching T2 geometry.
    """
    def __init__(self, keys: KeysCollection, t2_key: str = "t2", adc_key: str = "adc", highb_key: str = "highb", output_key: str = "image"):
        super().__init__(keys)
        self.t2_key = t2_key
        self.adc_key = adc_key
        self.highb_key = highb_key
        self.output_key = output_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        # Get T2 (required)
        t2 = d[self.t2_key]
        if isinstance(t2, MetaTensor):
            t2_data = t2.as_tensor() if hasattr(t2, 'as_tensor') else t2
        else:
            t2_data = torch.as_tensor(t2) if isinstance(t2, np.ndarray) else t2
        
        # Ensure T2 is [C, H, W, D] format
        if t2_data.dim() == 3:
            t2_data = t2_data.unsqueeze(0)
        if t2_data.dim() == 4 and t2_data.shape[0] == 1:
            pass  # Already [1, H, W, D]
        elif t2_data.dim() == 4:
            t2_data = t2_data.unsqueeze(0)  # [H, W, D] -> [1, H, W, D]
        
        spatial_shape = t2_data.shape[1:]  # [H, W, D]
        
        # Get ADC (optional)
        adc_data = None
        if self.adc_key in d and d[self.adc_key] is not None:
            # Skip if empty string
            if isinstance(d[self.adc_key], str) and not d[self.adc_key].strip():
                pass
            else:
                adc = d[self.adc_key]
                if isinstance(adc, MetaTensor):
                    adc_data = adc.as_tensor() if hasattr(adc, 'as_tensor') else adc
                else:
                    adc_data = torch.as_tensor(adc) if isinstance(adc, np.ndarray) else adc
                
                if adc_data.dim() == 3:
                    adc_data = adc_data.unsqueeze(0)
                if adc_data.shape[1:] != spatial_shape:
                    # Resize to match T2
                    adc_np = adc_data.cpu().numpy() if torch.is_tensor(adc_data) else adc_data
                    adc_resized = resize(adc_np[0], output_shape=spatial_shape, order=1)
                    adc_data = torch.as_tensor(adc_resized).unsqueeze(0) if torch.is_tensor(adc_data) else np.expand_dims(adc_resized, 0)
        
        if adc_data is None:
            # Create zeros matching T2
            if torch.is_tensor(t2_data):
                adc_data = torch.zeros_like(t2_data)
            else:
                adc_data = np.zeros_like(t2_data)
        
        # Get HIGHB (optional)
        highb_data = None
        if self.highb_key in d and d[self.highb_key] is not None:
            # Skip if empty string
            if isinstance(d[self.highb_key], str) and not d[self.highb_key].strip():
                pass
            else:
                highb = d[self.highb_key]
                if isinstance(highb, MetaTensor):
                    highb_data = highb.as_tensor() if hasattr(highb, 'as_tensor') else highb
                else:
                    highb_data = torch.as_tensor(highb) if isinstance(highb, np.ndarray) else highb
                
                if highb_data.dim() == 3:
                    highb_data = highb_data.unsqueeze(0)
                if highb_data.shape[1:] != spatial_shape:
                    # Resize to match T2
                    highb_np = highb_data.cpu().numpy() if torch.is_tensor(highb_data) else highb_data
                    highb_resized = resize(highb_np[0], output_shape=spatial_shape, order=1)
                    highb_data = torch.as_tensor(highb_resized).unsqueeze(0) if torch.is_tensor(highb_data) else np.expand_dims(highb_resized, 0)
        
        if highb_data is None:
            # Create zeros matching T2
            if torch.is_tensor(t2_data):
                highb_data = torch.zeros_like(t2_data)
            else:
                highb_data = np.zeros_like(t2_data)
        
        # Stack into 3-channel image [3, H, W, D]
        if torch.is_tensor(t2_data):
            stacked = torch.cat([t2_data, adc_data, highb_data], dim=0)
        else:
            stacked = np.concatenate([t2_data, adc_data, highb_data], axis=0)
        
        # Preserve metadata from T2 if available
        if isinstance(t2, MetaTensor):
            stacked = MetaTensor(stacked, meta=t2.meta)
        
        d[self.output_key] = stacked
        
        # Remove individual modality keys
        d.pop(self.t2_key, None)
        d.pop(self.adc_key, None)
        d.pop(self.highb_key, None)
        
        return d


def bbox2_3D(img: np.ndarray) -> list:
    """
    Computes 3D bounding box of non-zero voxels.
    Returns [rmin, rmax, cmin, cmax, zmin, zmax]
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    
    rmin, rmax = np.where(r)[0][[0, -1]] if np.any(r) else (0, img.shape[0] - 1)
    cmin, cmax = np.where(c)[0][[0, -1]] if np.any(c) else (0, img.shape[1] - 1)
    zmin, zmax = np.where(z)[0][[0, -1]] if np.any(z) else (0, img.shape[2] - 1)
    
    return [rmin, rmax, cmin, cmax, zmin, zmax]


class CropROIFromOrganMask(MapTransform):
    """
    Crops image and label to ROI computed from organ mask with margin.
    This matches the inference preprocessing in custom_lesion_seg_operator.py
    """
    def __init__(
        self,
        keys: KeysCollection,
        organ_mask_key: str = "organ_mask",
        margin: int = 32,
        image_key: str = "image",
    ):
        super().__init__(keys)
        self.organ_mask_key = organ_mask_key
        self.margin = margin
        self.image_key = image_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        # Get organ mask
        organ_mask = d[self.organ_mask_key]
        if isinstance(organ_mask, MetaTensor):
            organ_mask_np = organ_mask.as_tensor().cpu().numpy() if torch.is_tensor(organ_mask.as_tensor()) else organ_mask.as_tensor()
        elif torch.is_tensor(organ_mask):
            organ_mask_np = organ_mask.cpu().numpy()
        else:
            organ_mask_np = np.asarray(organ_mask)
        
        # Squeeze channel dimension if present
        if organ_mask_np.ndim == 4 and organ_mask_np.shape[0] == 1:
            organ_mask_np = organ_mask_np[0]
        elif organ_mask_np.ndim == 5 and organ_mask_np.shape[0] == 1 and organ_mask_np.shape[1] == 1:
            organ_mask_np = organ_mask_np[0, 0]
        
        # Binarize organ mask
        organ_mask_binary = (organ_mask_np > 0.0).astype(np.float32)
        
        # Compute bounding box
        bbox = bbox2_3D(organ_mask_binary)
        bbox_new = np.array(bbox)
        
        # Add margin
        spatial_shape = organ_mask_binary.shape
        for i in range(3):
            bbox_new[2*i] = max(0, bbox[2*i] - self.margin)
            bbox_new[2*i + 1] = min(spatial_shape[i] - 1, bbox[2*i + 1] + self.margin)
        
        # Crop all keys
        for key in self.keys:
            item = d[key]
            if isinstance(item, MetaTensor):
                item_data = item.as_tensor() if hasattr(item, 'as_tensor') else item
            elif torch.is_tensor(item):
                item_data = item
            else:
                item_data = np.asarray(item)
            
            # Handle different tensor shapes
            if item_data.ndim == 4:  # [C, H, W, D]
                cropped = item_data[
                    :,
                    bbox_new[0]:bbox_new[1] + 1,
                    bbox_new[2]:bbox_new[3] + 1,
                    bbox_new[4]:bbox_new[5] + 1
                ]
            elif item_data.ndim == 3:  # [H, W, D]
                cropped = item_data[
                    bbox_new[0]:bbox_new[1] + 1,
                    bbox_new[2]:bbox_new[3] + 1,
                    bbox_new[4]:bbox_new[5] + 1
                ]
            else:
                cropped = item_data  # Skip cropping if shape doesn't match
            
            # Preserve metadata if MetaTensor
            if isinstance(item, MetaTensor):
                d[key] = MetaTensor(cropped, meta=item.meta)
            else:
                d[key] = cropped
        
        # Remove organ_mask key (no longer needed after cropping)
        d.pop(self.organ_mask_key, None)
        
        return d


class MultiChannelNormalize(MapTransform):
    """
    Per-channel z-score normalization for multi-channel images (T2, ADC, HIGHB).
    Matches standard_normalization_multi_channel from custom_lesion_seg_operator.py
    """
    def __init__(self, keys: KeysCollection, nonzero: bool = True):
        super().__init__(keys)
        self.nonzero = nonzero

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        
        for key in self.keys:
            item = d[key]
            if isinstance(item, MetaTensor):
                item_data = item.as_tensor() if hasattr(item, 'as_tensor') else item
            elif torch.is_tensor(item):
                item_data = item
            else:
                item_data = np.asarray(item)
            
            # Convert to numpy for normalization
            if torch.is_tensor(item_data):
                item_np = item_data.cpu().numpy()
                was_tensor = True
            else:
                item_np = item_data
                was_tensor = False
            
            # Normalize each channel independently
            normalized = np.zeros_like(item_np)
            for c in range(item_np.shape[0]):
                channel = item_np[c]
                if self.nonzero:
                    mask = channel > 0
                    if np.any(mask):
                        mean = np.mean(channel[mask])
                        std = np.std(channel[mask])
                        if std > 0:
                            normalized[c] = (channel - mean) / std
                        else:
                            normalized[c] = channel - mean
                    else:
                        normalized[c] = channel
                else:
                    mean = np.mean(channel)
                    std = np.std(channel)
                    if std > 0:
                        normalized[c] = (channel - mean) / std
                    else:
                        normalized[c] = channel
            
            # Convert back to tensor if needed
            if was_tensor:
                normalized = torch.as_tensor(normalized, dtype=item_data.dtype)
            
            # Preserve metadata if MetaTensor
            if isinstance(item, MetaTensor):
                d[key] = MetaTensor(normalized, meta=item.meta)
            else:
                d[key] = normalized
        
        return d


def get_lesion_train_transforms(
    spacing: Sequence[float],
    roi_size: Sequence[int],
    roi_margin: int,
    samples_per_image: int,
    pos_to_neg: float,
):
    """
    Training transforms for binary lesion segmentation (background=0, lesion=1).
    - Loads T2, ADC, HIGHB and stacks into 3-channel image
    - Resamples to 0.5mm spacing
    - Crops ROI from organ mask with margin
    - Multi-channel normalization
    - Random spatial cropping and augmentations
    """
    return Compose(
        [
            DropNoneKeys(keys=["label"]),
            LoadImaged(keys=["t2", "organ_mask"]),
            LoadImaged(keys=["label"], allow_missing_keys=True),  # Lesion mask is optional
            LoadImaged(keys=["adc", "highb"], allow_missing_keys=True),
            # Create empty label if missing (no tumor detected)
            CreateEmptyLabelIfMissing(keys=["label"], image_key="t2"),
            EnsureChannelFirstd(keys=["t2", "adc", "highb", "organ_mask", "label"]),
            Orientationd(keys=["t2", "adc", "highb", "organ_mask", "label"], axcodes="RAS"),
            # Resample to target spacing (0.5mm)
            Spacingd(
                keys=["t2", "adc", "highb", "organ_mask", "label"],
                pixdim=spacing,
                mode=("bilinear", "bilinear", "bilinear", "nearest", "nearest"),
            ),
            # Stack multi-modal images into 3-channel
            StackMultiModalImages(
                keys=["t2", "adc", "highb"],
                t2_key="t2",
                adc_key="adc",
                highb_key="highb",
                output_key="image",
            ),
            # Ensure label matches image size
            ResizeLabelToMatchImage(keys=["label"], image_key="image", mode="nearest"),
            ResizeLabelToMatchImage(keys=["organ_mask"], image_key="image", mode="nearest"),
            # Crop ROI from organ mask
            CropROIFromOrganMask(
                keys=["image", "label"],
                organ_mask_key="organ_mask",
                margin=roi_margin,
                image_key="image",
            ),
            # Pad to ensure we can take fixed-size crops even if ROI is smaller than roi_size
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="constant"),
            # Multi-channel normalization
            MultiChannelNormalize(keys=["image"], nonzero=True),
            # Random spatial cropping
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=roi_size,
                num_samples=samples_per_image,
                random_size=False,
            ),
            # Augmentations
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                prob=0.15,
                rotate_range=(0.0, 0.0, 0.261799),  # ~15 degrees around z
                scale_range=(0.1, 0.1, 0.0),
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_lesion_val_transforms(
    spacing: Sequence[float],
    roi_margin: int,
    roi_size: Sequence[int],
):
    """
    Validation transforms for binary lesion segmentation (background=0, lesion=1).
    - Loads T2, ADC, HIGHB and stacks into 3-channel image
    - Resamples to 0.5mm spacing
    - Crops ROI from organ mask with margin
    - Multi-channel normalization
    """
    return Compose(
        [
            DropNoneKeys(keys=["label"]),
            LoadImaged(keys=["t2", "organ_mask"]),
            LoadImaged(keys=["label"], allow_missing_keys=True),  # Lesion mask is optional
            LoadImaged(keys=["adc", "highb"], allow_missing_keys=True),
            # Create empty label if missing (no tumor detected)
            CreateEmptyLabelIfMissing(keys=["label"], image_key="t2"),
            EnsureChannelFirstd(keys=["t2", "adc", "highb", "organ_mask", "label"]),
            Orientationd(keys=["t2", "adc", "highb", "organ_mask", "label"], axcodes="RAS"),
            # Resample to target spacing (0.5mm)
            Spacingd(
                keys=["t2", "adc", "highb", "organ_mask", "label"],
                pixdim=spacing,
                mode=("bilinear", "bilinear", "bilinear", "nearest", "nearest"),
            ),
            # Stack multi-modal images into 3-channel
            StackMultiModalImages(
                keys=["t2", "adc", "highb"],
                t2_key="t2",
                adc_key="adc",
                highb_key="highb",
                output_key="image",
            ),
            # Ensure label matches image size
            ResizeLabelToMatchImage(keys=["label"], image_key="image", mode="nearest"),
            ResizeLabelToMatchImage(keys=["organ_mask"], image_key="image", mode="nearest"),
            # Crop ROI from organ mask
            CropROIFromOrganMask(
                keys=["image", "label"],
                organ_mask_key="organ_mask",
                margin=roi_margin,
                image_key="image",
            ),
            # Pad to ensure fixed-size evaluation crops
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="constant"),
            # Multi-channel normalization
            MultiChannelNormalize(keys=["image"], nonzero=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )