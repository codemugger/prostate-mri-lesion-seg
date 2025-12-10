from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from monai.data import Dataset, CacheDataset
import nibabel as nib
import numpy as np
import SimpleITK as sitk


@dataclass
class OrganDatasetPaths:
    t2_dir: str
    label_dir: str


@dataclass
class LesionDatasetPaths:
    t2_dir: str
    adc_dir: str
    highb_dir: str
    organ_mask_dir: str
    lesion_mask_dir: str


def read_subject_ids(csv_path: str) -> List[str]:
    """
    Reads a CSV with a single column `subject_id` and returns the list of IDs.
    """
    subject_ids: List[str] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "subject_id" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'subject_id' column.")
        for row in reader:
            sid = row["subject_id"].strip()
            if sid:
                subject_ids.append(sid)
    if not subject_ids:
        raise ValueError(f"No subject_id found in {csv_path}")
    return subject_ids


def _extract_base_id_from_t2_filename(filename: str) -> str:
    """
    Extracts the base subject ID from a T2 filename that may contain suffixes.
    Example:
      - ProstateX-0004_t2_tse_tra_5.nii.gz -> ProstateX-0004
      - ProstateX-0004.nii.gz -> ProstateX-0004
    """
    if not filename.endswith(".nii.gz"):
        return ""
    stem = filename[:-7]  # strip .nii.gz
    # split on the first underscore if present
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def list_subject_ids_from_dirs(t2_dir: str, label_dir: str) -> List[str]:
    """
    Lists subject IDs by intersecting filename stems present in both t2_dir and label_dir.
    T2 filenames may have suffixes; labels are assumed to be <SUBJECT_ID>.nii.gz
    """
    def t2_base_ids(dir_path: str) -> set:
        result = set()
        for name in os.listdir(dir_path):
            if not name.endswith(".nii.gz"):
                continue
            base = _extract_base_id_from_t2_filename(name)
            if base:
                result.add(base)
        return result

    def label_stems(dir_path: str) -> set:
        result = set()
        for name in os.listdir(dir_path):
            if not name.endswith(".nii.gz"):
                continue
            stem = name[:-7]
            if stem:
                result.add(stem)
        return result

    if not os.path.isdir(t2_dir):
        raise FileNotFoundError(f"Not a directory: {t2_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Not a directory: {label_dir}")

    img_stems = t2_base_ids(t2_dir)
    lbl_stems = label_stems(label_dir)
    common = sorted(list(img_stems.intersection(lbl_stems)))
    if not common:
        raise ValueError(f"No common subject IDs found between {t2_dir} and {label_dir}")
    return common


def list_lesion_subject_ids_from_dirs(
    t2_dir: str,
    organ_mask_dir: str,
    lesion_mask_dir: str,
) -> List[str]:
    """
    Lists subject IDs that have T2 and organ mask.
    Lesion mask is optional - missing masks represent cases with no tumor detected.
    """
    def t2_base_ids(dir_path: str) -> set:
        result = set()
        for name in os.listdir(dir_path):
            if not name.endswith(".nii.gz"):
                continue
            base = _extract_base_id_from_t2_filename(name)
            if base:
                result.add(base)
        return result

    def mask_stems(dir_path: str) -> set:
        result = set()
        if not os.path.isdir(dir_path):
            return result
        for name in os.listdir(dir_path):
            if not name.endswith(".nii.gz"):
                continue
            stem = name[:-7]  # Remove .nii.gz
            if stem:
                result.add(stem)
        return result

    if not os.path.isdir(t2_dir):
        raise FileNotFoundError(f"Not a directory: {t2_dir}")
    if not os.path.isdir(organ_mask_dir):
        raise FileNotFoundError(f"Not a directory: {organ_mask_dir}")
    # lesion_mask_dir is optional - may not exist or may be empty

    t2_ids = t2_base_ids(t2_dir)
    organ_ids = mask_stems(organ_mask_dir)
    
    # Require T2 and organ mask (lesion mask is optional)
    common = sorted(list(t2_ids.intersection(organ_ids)))
    if not common:
        raise ValueError(
            f"No common subject IDs found across T2 ({t2_dir}) and organ masks ({organ_mask_dir})"
        )
    return common


def _resolve_t2_path_for_subject(t2_dir: str, subject_id: str) -> str:
    """
    Resolves the actual T2 file path for a subject, supporting suffixed filenames.
    Preference:
      1) Exact match <SUBJECT_ID>.nii.gz
      2) File starting with <SUBJECT_ID>_t2 (case-insensitive)
      3) Any file starting with <SUBJECT_ID>_
      4) Otherwise raise FileNotFoundError
    """
    # Normalize subject_id: remove .nii extension if present (CSV files may have it)
    base_id = subject_id.replace(".nii", "").replace(".gz", "")
    
    exact = os.path.join(t2_dir, f"{base_id}.nii.gz")
    if os.path.exists(exact):
        return exact
    candidates = []
    t2_pref = []
    prefix = f"{base_id}_"
    for name in os.listdir(t2_dir):
        if not name.endswith(".nii.gz"):
            continue
        if name.startswith(prefix):
            candidates.append(name)
            lower = name.lower()
            if "_t2" in lower:
                t2_pref.append(name)
    chosen = None
    if t2_pref:
        chosen = sorted(t2_pref)[0]
    elif candidates:
        chosen = sorted(candidates)[0]
    if chosen is None:
        raise FileNotFoundError(f"No T2 file found for subject {subject_id} in {t2_dir}")
    return os.path.join(t2_dir, chosen)


def build_organ_items(subject_ids: List[str], paths: OrganDatasetPaths) -> List[Dict[str, str]]:
    """
    Builds a list of dicts with MONAI dictionary keys for T2 image and organ mask label.
    """
    items: List[Dict[str, str]] = []
    for sid in subject_ids:
        # Normalize subject_id: remove .nii extension if present (CSV files may have it)
        base_id = sid.replace(".nii", "").replace(".gz", "")
        
        img_path = _resolve_t2_path_for_subject(paths.t2_dir, sid)
        label_path = os.path.join(paths.label_dir, f"{base_id}.nii.gz")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing label: {label_path}")
        items.append({"image": img_path, "label": label_path, "subject_id": base_id})
    return items


def create_monai_datasets(
    train_items: List[Dict[str, str]],
    val_items: List[Dict[str, str]],
    train_transforms,
    val_transforms,
    cache_rate: float,
    num_workers: int,
) -> Tuple[Dataset, Dataset]:
    """
    Creates MONAI CacheDataset for train/val for faster IO and transform caching.
    """
    train_ds = CacheDataset(train_items, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    val_ds = CacheDataset(val_items, transform=val_transforms, cache_rate=0.0, num_workers=num_workers)
    return train_ds, val_ds


def _resolve_modality_path(modality_dir: str, subject_id: str, modality: str) -> Optional[str]:
    """
    Resolves the path for a modality (ADC or HIGHB) for a subject.
    Returns None if not found (modalities are optional).
    """
    base_id = subject_id.replace(".nii", "").replace(".gz", "")
    
    # Try common naming patterns
    patterns = [
        f"{base_id}_{modality}.nii.gz",
        f"{base_id}_{modality.upper()}.nii.gz",
        f"{modality}_{base_id}.nii.gz",
        f"{modality.upper()}_{base_id}.nii.gz",
        f"{base_id}.nii.gz",  # Fallback: exact match
    ]
    
    for pattern in patterns:
        path = os.path.join(modality_dir, pattern)
        if os.path.exists(path):
            return path
    
    return None


def _resample_to_match_t2(adc_path: Optional[str], highb_path: Optional[str], t2_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resamples ADC and HIGHB to match T2 geometry using SimpleITK.
    Returns paths to resampled files (or None if input was None or empty string).
    """
    t2_sitk = sitk.ReadImage(t2_path)
    t2_size = t2_sitk.GetSize()
    t2_spacing = t2_sitk.GetSpacing()
    t2_origin = t2_sitk.GetOrigin()
    t2_direction = t2_sitk.GetDirection()
    t2_pixel_id = t2_sitk.GetPixelID()
    
    os.makedirs(output_dir, exist_ok=True)
    
    adc_resampled = None
    if adc_path and adc_path.strip() and os.path.exists(adc_path):
        try:
            adc_sitk = sitk.ReadImage(adc_path)
            adc_resampled_sitk = sitk.Resample(
                adc_sitk, t2_size,
                sitk.Transform(),
                sitk.sitkLinear,  # Use linear interpolation for ADC
                t2_origin,
                t2_spacing,
                t2_direction,
                0,
                t2_pixel_id
            )
            adc_output_path = os.path.join(output_dir, os.path.basename(adc_path))
            sitk.WriteImage(adc_resampled_sitk, adc_output_path)
            adc_resampled = adc_output_path
        except Exception as e:
            print(f"Warning: Failed to resample ADC {adc_path}: {e}")
            adc_resampled = None
    
    highb_resampled = None
    if highb_path and highb_path.strip() and os.path.exists(highb_path):
        try:
            highb_sitk = sitk.ReadImage(highb_path)
            highb_resampled_sitk = sitk.Resample(
                highb_sitk, t2_size,
                sitk.Transform(),
                sitk.sitkLinear,  # Use linear interpolation for HIGHB
                t2_origin,
                t2_spacing,
                t2_direction,
                0,
                t2_pixel_id
            )
            highb_output_path = os.path.join(output_dir, os.path.basename(highb_path))
            sitk.WriteImage(highb_resampled_sitk, highb_output_path)
            highb_resampled = highb_output_path
        except Exception as e:
            print(f"Warning: Failed to resample HIGHB {highb_path}: {e}")
            highb_resampled = None
    
    return adc_resampled, highb_resampled


def build_lesion_items(subject_ids: List[str], paths: LesionDatasetPaths, temp_resample_dir: str = "/tmp/lesion_resampled") -> List[Dict[str, str]]:
    """
    Builds a list of dicts with MONAI dictionary keys for multi-parametric lesion segmentation.
    Keys: "image" (stacked T2+ADC+HIGHB), "organ_mask", "label" (lesion mask), "subject_id"
    
    Note: ADC and HIGHB are optional. If missing, they will be set to zeros matching T2 geometry.
    """
    items: List[Dict[str, str]] = []
    
    for sid in subject_ids:
        base_id = sid.replace(".nii", "").replace(".gz", "")
        
        # Resolve T2 path (required)
        t2_path = _resolve_t2_path_for_subject(paths.t2_dir, sid)
        if not os.path.exists(t2_path):
            raise FileNotFoundError(f"T2 file not found for subject {sid}: {t2_path}")
        
        # Resolve ADC and HIGHB paths (optional)
        adc_path = _resolve_modality_path(paths.adc_dir, sid, "adc")
        highb_path = _resolve_modality_path(paths.highb_dir, sid, "highb")
        
        # Resample ADC and HIGHB to match T2 geometry
        adc_resampled, highb_resampled = _resample_to_match_t2(
            adc_path, highb_path, t2_path, temp_resample_dir
        )
        
        # Resolve organ mask path (required for ROI cropping)
        organ_mask_path = os.path.join(paths.organ_mask_dir, f"{base_id}.nii.gz")
        if not os.path.exists(organ_mask_path):
            raise FileNotFoundError(f"Organ mask not found for subject {sid}: {organ_mask_path}")
        
        # Resolve lesion mask path (optional - missing means no tumor detected)
        lesion_mask_path = os.path.join(paths.lesion_mask_dir, f"{base_id}.nii.gz")
        if not os.path.exists(lesion_mask_path):
            # If lesion mask doesn't exist, we'll create an empty mask (all zeros) during loading
            # This represents cases where no tumor was detected
            lesion_mask_path = None
        
        item = {
            "t2": t2_path,
            "organ_mask": organ_mask_path,
            "label": lesion_mask_path,
            "subject_id": base_id,
        }
        # Only add ADC/HIGHB if they exist (optional modalities)
        if adc_resampled:
            item["adc"] = adc_resampled
        if highb_resampled:
            item["highb"] = highb_resampled
        
        items.append(item)
    
    return items
