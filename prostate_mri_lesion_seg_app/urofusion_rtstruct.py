"""UroFusion-compatible canonical RTSTRUCT export.

The production entry point :func:`generate_accepted_rtstruct` creates the
radiologist- and vendor-accepted, non-smoothed combined RTSTRUCT from the
patient-specific NIfTI masks.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    RTStructureSetStorage,
    UID,
    generate_uid,
)
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
import yaml

from rt_utils import RTStructBuilder


CANONICAL_RTSTRUCT_NAME = "combined_organ_lesion_RTSTRUCT.dcm"

REQUIRED_CASE_PATHS = (
    Path("organ/cleaned_organ.nii.gz"),
    Path("lesion/lesion_mask.nii.gz"),
    Path("lesions.txt"),
    Path("dicom/t2"),
)

LESION_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 200, 255),
    (255, 215, 0),
    (180, 80, 255),
    (0, 220, 160),
    (255, 80, 120),
)

PATIENT_STUDY_KEYWORDS = (
    "SpecificCharacterSet",
    "PatientName",
    "PatientID",
    "IssuerOfPatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "ReferringPhysicianName",
    "StudyDescription",
    "StudyInstanceUID",
    "PatientIdentityRemoved",
    "DeidentificationMethod",
    "DeidentificationMethodCodeSequence",
)

# These RT Structure Set patient/study/equipment attributes are Type 2: the
# attribute must exist, while a zero-length value truthfully represents an
# absent source value. No placeholder identifiers are generated.
REQUIRED_TYPE2_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "StudyID",
    "AccessionNumber",
    "SeriesNumber",
    "StructureSetDate",
    "StructureSetTime",
    "Manufacturer",
    "OperatorsName",
    "PositionReferenceIndicator",
)

SOURCE_TYPE2_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "StudyID",
    "AccessionNumber",
)

DICOM_TEXT_VR_MAX_LENGTH = {
    "AE": 16,
    "AS": 4,
    "CS": 16,
    "DA": 8,
    "DS": 16,
    "DT": 26,
    "IS": 12,
    "LO": 64,
    "LT": 10240,
    "PN": 64,
    "SH": 16,
    "ST": 1024,
    "TM": 16,
    "UI": 64,
}


class ReviewExportError(RuntimeError):
    """Base exception for a canonical RTSTRUCT export failure."""


class IneligibleCaseError(ReviewExportError):
    """Raised when a case lacks prerequisites required for a truthful export."""


class GeometryMismatchError(ReviewExportError):
    """Raised when NIfTI and source-DICOM geometry do not agree exactly."""


@dataclass(frozen=True)
class RTStructConfig:
    max_volume_delta_percent: float = 8.0
    grid_tolerance_mm: float = 1e-3
    affine_tolerance_mm: float = 1e-3


def _validate_export_config(config: RTStructConfig) -> None:
    if not 0.0 <= config.max_volume_delta_percent <= 8.0:
        raise ValueError("Volume gate may be tightened but never exceed 8 percent")
    for name, tolerance in (
        ("grid_tolerance_mm", config.grid_tolerance_mm),
        ("affine_tolerance_mm", config.affine_tolerance_mm),
    ):
        if not 0.0 < tolerance <= 1e-2:
            raise ValueError(f"{name} must be greater than 0 and at most 0.01 mm")


@dataclass(frozen=True)
class DicomSlice:
    path: Path
    header: Dataset
    position: float


@dataclass
class SourceSeries:
    slices: list[DicomSlice]
    ignored_files: list[dict[str, str]] = field(default_factory=list)

    @property
    def first(self) -> Dataset:
        return self.slices[0].header

    @property
    def sop_instance_uids(self) -> list[str]:
        return [str(item.header.SOPInstanceUID) for item in self.slices]

    @property
    def study_instance_uid(self) -> str:
        return str(self.first.StudyInstanceUID)

    @property
    def series_instance_uid(self) -> str:
        return str(self.first.SeriesInstanceUID)

    @property
    def frame_of_reference_uid(self) -> str:
        return str(self.first.FrameOfReferenceUID)

    @property
    def rows(self) -> int:
        return int(self.first.Rows)

    @property
    def columns(self) -> int:
        return int(self.first.Columns)

    @property
    def row_direction(self) -> np.ndarray:
        return np.asarray(self.first.ImageOrientationPatient[:3], dtype=float)

    @property
    def column_direction(self) -> np.ndarray:
        return np.asarray(self.first.ImageOrientationPatient[3:], dtype=float)

    @property
    def normal(self) -> np.ndarray:
        value = np.cross(self.row_direction, self.column_direction)
        return value / np.linalg.norm(value)

    @property
    def pixel_spacing(self) -> tuple[float, float]:
        values = tuple(float(value) for value in self.first.PixelSpacing)
        return values[0], values[1]

    @property
    def slice_spacing(self) -> float:
        if len(self.slices) > 1:
            differences = np.diff([item.position for item in self.slices])
            return float(np.median(differences))
        return float(
            getattr(
                self.first,
                "SpacingBetweenSlices",
                getattr(self.first, "SliceThickness", 1.0),
            )
        )

    @property
    def expected_ras_affine(self) -> np.ndarray:
        """Return the source single-frame MR grid as a NIfTI RAS affine."""
        row_spacing, column_spacing = self.pixel_spacing
        lps_affine = np.eye(4, dtype=float)
        # NIfTI axis 0 advances through DICOM columns; axis 1 through rows.
        lps_affine[:3, 0] = self.row_direction * column_spacing
        lps_affine[:3, 1] = self.column_direction * row_spacing
        lps_affine[:3, 2] = self.normal * self.slice_spacing
        lps_affine[:3, 3] = np.asarray(
            self.slices[0].header.ImagePositionPatient,
            dtype=float,
        )
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        return lps_to_ras @ lps_affine


@dataclass
class SingleContourProfileResult:
    mask: np.ndarray
    accepted: bool
    metrics: dict[str, Any]
    failed_gates: list[str]


@dataclass
class LesionROI:
    display_id: int
    native_component_id: int
    classifier_component_id: int | None
    pi_rads: int | None
    match_status: str
    mask: np.ndarray

    @property
    def name(self) -> str:
        score = str(self.pi_rads) if self.pi_rads is not None else "unavailable"
        return f"Lesion {self.display_id} - AI PI-RADS {score}"

    @property
    def observation_label(self) -> str:
        return f"LESION_{self.display_id}"[:16]


def case_missing_prerequisites(case_directory: Path) -> list[str]:
    case_directory = Path(case_directory)
    missing: list[str] = []
    for relative_path in REQUIRED_CASE_PATHS:
        path = case_directory / relative_path
        if relative_path == Path("dicom/t2"):
            if not path.is_dir() or not any(path.glob("*.dcm")):
                missing.append(str(relative_path))
        elif not path.is_file():
            missing.append(str(relative_path))
    return missing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source_series(
    directory: Path,
    *,
    grid_tolerance_mm: float = 1e-3,
) -> SourceSeries:
    directory = Path(directory)
    if not directory.is_dir():
        raise IneligibleCaseError(f"Missing source T2 DICOM directory: {directory}")

    image_headers: list[tuple[Path, Dataset]] = []
    ignored: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.dcm")):
        header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        sop_class = str(getattr(header, "SOPClassUID", ""))
        has_spatial_plane = all(
            hasattr(header, keyword)
            for keyword in (
                "ImageOrientationPatient",
                "ImagePositionPatient",
                "PixelSpacing",
                "Rows",
                "Columns",
                "SOPInstanceUID",
            )
        )
        if sop_class != str(MRImageStorage) or not has_spatial_plane:
            ignored.append(
                {
                    "file": path.name,
                    "sop_class_uid": sop_class,
                    "reason": "not_single_frame_spatial_mr_image",
                    "sha256": _sha256(path),
                }
            )
            continue
        image_headers.append((path, header))

    if not image_headers:
        raise IneligibleCaseError(
            f"No single-frame spatial MR Image Storage objects found in {directory}"
        )

    first = image_headers[0][1]
    for required in (
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "FrameOfReferenceUID",
    ):
        if not getattr(first, required, ""):
            raise GeometryMismatchError(f"Source T2 is missing required {required}")

    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    if orientation.shape != (6,) or not np.all(np.isfinite(orientation)):
        raise GeometryMismatchError("Invalid source ImageOrientationPatient")
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    if not np.isclose(np.linalg.norm(row_direction), 1.0, atol=1e-5):
        raise GeometryMismatchError("Source row direction cosine is not unit length")
    if not np.isclose(np.linalg.norm(column_direction), 1.0, atol=1e-5):
        raise GeometryMismatchError("Source column direction cosine is not unit length")
    if not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-5):
        raise GeometryMismatchError("Source DICOM orientation axes are not orthogonal")
    normal = np.cross(row_direction, column_direction)
    normal /= np.linalg.norm(normal)

    study_uid = str(first.StudyInstanceUID)
    series_uid = str(first.SeriesInstanceUID)
    frame_uid = str(first.FrameOfReferenceUID)
    rows = int(first.Rows)
    columns = int(first.Columns)
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    seen_sops: set[str] = set()
    positioned: list[DicomSlice] = []
    for path, header in image_headers:
        if str(header.StudyInstanceUID) != study_uid:
            raise GeometryMismatchError("T2 directory contains multiple studies")
        if str(header.SeriesInstanceUID) != series_uid:
            raise GeometryMismatchError("T2 directory contains multiple image series")
        if str(header.FrameOfReferenceUID) != frame_uid:
            raise GeometryMismatchError("T2 images use multiple Frames of Reference")
        if int(header.Rows) != rows or int(header.Columns) != columns:
            raise GeometryMismatchError("T2 image matrix dimensions are inconsistent")
        if not np.allclose(
            np.asarray(header.ImageOrientationPatient, dtype=float),
            orientation,
            rtol=0.0,
            atol=1e-5,
        ):
            raise GeometryMismatchError("T2 image orientations are inconsistent")
        if not np.allclose(
            np.asarray(header.PixelSpacing, dtype=float),
            pixel_spacing,
            rtol=0.0,
            atol=grid_tolerance_mm,
        ):
            raise GeometryMismatchError("T2 pixel spacings are inconsistent")
        sop_uid = str(header.SOPInstanceUID)
        if sop_uid in seen_sops:
            raise GeometryMismatchError(f"Duplicate T2 SOP Instance UID: {sop_uid}")
        seen_sops.add(sop_uid)
        position = float(
            np.dot(np.asarray(header.ImagePositionPatient, dtype=float), normal)
        )
        positioned.append(DicomSlice(path=path, header=header, position=position))

    positioned.sort(key=lambda item: item.position)
    positions = np.asarray([item.position for item in positioned], dtype=float)
    if len(positions) > 1 and np.any(np.diff(positions) <= grid_tolerance_mm):
        raise GeometryMismatchError("T2 series contains duplicate slice positions")
    if len(positions) > 2:
        differences = np.diff(positions)
        median = float(np.median(differences))
        if median <= 0.0 or not np.allclose(
            differences,
            median,
            rtol=1e-3,
            atol=grid_tolerance_mm,
        ):
            raise GeometryMismatchError("T2 slice spacing is nonuniform")

    return SourceSeries(slices=positioned, ignored_files=ignored)


def _validate_nifti_grid(
    image: nib.spatialimages.SpatialImage,
    series: SourceSeries,
    *,
    affine_tolerance_mm: float,
) -> None:
    expected_shape = (series.columns, series.rows, len(series.slices))
    if tuple(image.shape) != expected_shape:
        raise GeometryMismatchError(
            f"NIfTI shape {tuple(image.shape)} does not match T2 grid {expected_shape}"
        )
    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise GeometryMismatchError("NIfTI affine is invalid")
    if not np.allclose(
        affine,
        series.expected_ras_affine,
        rtol=0.0,
        atol=affine_tolerance_mm,
    ):
        difference = float(np.max(np.abs(affine - series.expected_ras_affine)))
        raise GeometryMismatchError(
            "NIfTI affine does not match the selected T2 patient grid "
            f"(maximum element difference {difference:.6f} mm)"
        )


def _validate_matching_nifti_grids(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    *,
    affine_tolerance_mm: float,
) -> None:
    if tuple(first.shape) != tuple(second.shape):
        raise GeometryMismatchError(
            f"Organ and lesion shapes differ: {first.shape} vs {second.shape}"
        )
    if not np.allclose(
        first.affine,
        second.affine,
        rtol=0.0,
        atol=affine_tolerance_mm,
    ):
        raise GeometryMismatchError("Organ and lesion NIfTI affines differ")


def _binary_mask(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise GeometryMismatchError(f"Expected a 3D NIfTI mask, got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise GeometryMismatchError("NIfTI mask contains non-finite values")
    return np.asarray(data > 0, dtype=bool)


def _planar_hole_summary(mask: np.ndarray) -> dict[str, Any]:
    by_slice: list[dict[str, int]] = []
    for slice_index in range(mask.shape[2]):
        plane = np.asarray(mask[:, :, slice_index], dtype=bool)
        holes = ndi.binary_fill_holes(plane) & ~plane
        voxel_count = int(np.count_nonzero(holes))
        if voxel_count:
            by_slice.append(
                {
                    "slice_index": slice_index,
                    "enclosed_hole_voxels": voxel_count,
                }
            )
    return {
        "affected_slice_count": len(by_slice),
        "enclosed_hole_voxel_count": int(
            sum(item["enclosed_hole_voxels"] for item in by_slice)
        ),
        "by_slice": by_slice,
    }


def _slice_component_summary(mask: np.ndarray) -> dict[str, Any]:
    by_slice: list[dict[str, int]] = []
    structure = ndi.generate_binary_structure(2, 1)
    for slice_index in range(mask.shape[2]):
        plane = np.asarray(mask[:, :, slice_index], dtype=bool)
        if not plane.any():
            continue
        _, component_count = ndi.label(plane, structure=structure)
        if component_count != 1:
            by_slice.append(
                {
                    "slice_index": slice_index,
                    "component_count_4": int(component_count),
                }
            )
    return {
        "affected_slice_count": len(by_slice),
        "by_slice": by_slice,
    }


def build_single_outer_prostate_profile(
    source_mask: np.ndarray,
    lesion_mask: np.ndarray,
    affine: np.ndarray,
    config: RTStructConfig = RTStructConfig(),
) -> SingleContourProfileResult:
    """Build one filled outer prostate region on each occupied T2 slice.

    This is a viewer-compatibility representation, not an anatomical
    correction. On every occupied slice it deterministically retains the
    largest 4-connected source component and fills only its enclosed 2D holes.
    Lesion voxels are inspected but are never changed or used to choose the
    retained prostate component.
    """
    source = np.asarray(source_mask, dtype=bool)
    lesion = np.asarray(lesion_mask, dtype=bool)
    if source.ndim != 3 or lesion.ndim != 3 or source.shape != lesion.shape:
        raise GeometryMismatchError(
            "Single-contour profiling requires matching 3D organ/lesion grids"
        )
    if not source.any():
        raise GeometryMismatchError("Cleaned prostate mask is empty")

    profile = np.zeros_like(source, dtype=bool)
    source_components_by_slice: list[dict[str, int]] = []
    retained_components_by_slice: list[dict[str, int]] = []
    removed_component_voxels = 0
    filled_hole_voxels = 0
    structure = ndi.generate_binary_structure(2, 1)
    for slice_index in range(source.shape[2]):
        plane = source[:, :, slice_index]
        if not plane.any():
            continue
        labels, component_count = ndi.label(plane, structure=structure)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        retained_component = int(np.argmax(sizes))
        largest = labels == retained_component
        filled = ndi.binary_fill_holes(largest)
        removed = int(np.count_nonzero(plane & ~largest))
        added = int(np.count_nonzero(filled & ~largest))
        removed_component_voxels += removed
        filled_hole_voxels += added
        profile[:, :, slice_index] = filled
        source_components_by_slice.append(
            {
                "slice_index": slice_index,
                "source_component_count_4": int(component_count),
                "retained_component_voxels": int(np.count_nonzero(largest)),
                "removed_component_voxels": removed,
                "filled_hole_voxels": added,
            }
        )
        if component_count > 1 or added:
            retained_components_by_slice.append(source_components_by_slice[-1])

    source_voxels = int(np.count_nonzero(source))
    profile_voxels = int(np.count_nonzero(profile))
    volume_delta_percent = 100.0 * (profile_voxels - source_voxels) / source_voxels
    source_slices = np.flatnonzero(np.any(source, axis=(0, 1)))
    profile_slices = np.flatnonzero(np.any(profile, axis=(0, 1)))
    profile_components = _slice_component_summary(profile)
    profile_holes = _planar_hole_summary(profile)
    lesion_outside_profile = int(np.count_nonzero(lesion & ~profile))
    voxel_volume_mm3 = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))

    failed: list[str] = []
    if not profile.any():
        failed.append("single_contour_profile_empty")
    if not np.array_equal(source_slices, profile_slices):
        failed.append("single_contour_profile_changed_occupied_slices")
    if profile_components["affected_slice_count"]:
        failed.append("single_contour_profile_has_multiple_slice_components")
    if profile_holes["affected_slice_count"]:
        failed.append("single_contour_profile_has_enclosed_slice_holes")
    if abs(volume_delta_percent) > config.max_volume_delta_percent:
        failed.append("single_contour_profile_volume_delta_exceeds_limit")

    metrics: dict[str, Any] = {
        "method": "largest_4_connected_component_then_fill_enclosed_holes_per_slice",
        "purpose": "UroFusion display compatibility; not anatomy correction",
        "source_voxels": source_voxels,
        "profile_voxels": profile_voxels,
        "changed_voxels": int(np.count_nonzero(source ^ profile)),
        "removed_component_voxels": removed_component_voxels,
        "filled_hole_voxels": filled_hole_voxels,
        "source_volume_mm3": float(source_voxels * voxel_volume_mm3),
        "profile_volume_mm3": float(profile_voxels * voxel_volume_mm3),
        "volume_delta_percent": float(volume_delta_percent),
        "maximum_absolute_volume_delta_percent": float(
            config.max_volume_delta_percent
        ),
        "source_occupied_slice_indices": [int(value) for value in source_slices],
        "profile_occupied_slice_indices": [int(value) for value in profile_slices],
        "source_slices_modified": retained_components_by_slice,
        "profile_multiple_component_summary": profile_components,
        "profile_planar_hole_summary": profile_holes,
        "lesion_voxels": int(np.count_nonzero(lesion)),
        "lesion_voxels_outside_profile": lesion_outside_profile,
        "accepted": not failed,
        "failed_gates": list(failed),
    }
    return SingleContourProfileResult(
        mask=profile,
        accepted=not failed,
        metrics=metrics,
        failed_gates=failed,
    )


def _parse_lesion_scores(path: Path) -> dict[int, int]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReviewExportError(f"Unable to parse {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, list):
        raise ReviewExportError("lesions.txt must contain a YAML list")
    scores: dict[int, int] = {}
    for item in payload:
        if not isinstance(item, dict) or "Lesion_ID" not in item:
            continue
        try:
            lesion_id = int(item["Lesion_ID"])
            score = int(item["PI_RADS"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewExportError("Invalid lesion ID or PI-RADS in lesions.txt") from exc
        if lesion_id <= 0 or score not in {2, 3, 4, 5}:
            raise ReviewExportError(
                f"Out-of-range lesion result: ID={lesion_id}, PI-RADS={score}"
            )
        if lesion_id in scores:
            raise ReviewExportError(f"Duplicate Lesion_ID {lesion_id} in lesions.txt")
        scores[lesion_id] = score
    return scores


def _classifier_grid_labels(
    native_mask: np.ndarray,
    native_affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the classifier's 0.5 mm nearest-neighbour lesion grid."""
    try:
        import torch
        from monai.data import MetaTensor
        from monai.transforms import Spacing

        tensor = MetaTensor(
            native_mask[np.newaxis].astype(np.uint8),
            affine=torch.as_tensor(native_affine, dtype=torch.float64),
        )
        resampled = Spacing(
            pixdim=(0.5, 0.5, 0.5),
            mode="nearest",
        )(tensor)
        resampled_mask = np.asarray(resampled[0].cpu().numpy() > 0, dtype=bool)
        resampled_affine = np.asarray(resampled.affine.cpu().numpy(), dtype=float)
    except Exception as exc:  # pragma: no cover - deployment dependency path
        raise ReviewExportError(
            f"Unable to reproduce classifier lesion grid: {exc}"
        ) from exc
    labels, _ = ndi.label(
        resampled_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    return labels.astype(np.int32), resampled_affine


def build_lesion_rois(
    lesion_mask: np.ndarray,
    affine: np.ndarray,
    scores: dict[int, int],
) -> tuple[list[LesionROI], dict[str, Any]]:
    """Map classifier-grid scores to native components using physical overlap."""
    native_labels, native_count = ndi.label(
        np.asarray(lesion_mask, dtype=bool),
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    classifier_labels, classifier_affine = _classifier_grid_labels(
        np.asarray(lesion_mask, dtype=bool),
        np.asarray(affine, dtype=float),
    )
    classifier_ids = [int(value) for value in np.unique(classifier_labels) if value]
    overlap: dict[int, dict[int, int]] = {}
    for classifier_id in classifier_ids:
        classifier_component = (classifier_labels == classifier_id).astype(np.uint8)
        back_resampled = np.asarray(
            resample_from_to(
                nib.Nifti1Image(classifier_component, classifier_affine),
                (native_labels.shape, affine),
                order=0,
            ).dataobj
        ) > 0
        counts: dict[int, int] = {}
        for native_id in np.unique(native_labels[back_resampled]):
            native_id = int(native_id)
            if native_id <= 0:
                continue
            counts[native_id] = int(
                np.count_nonzero(back_resampled & (native_labels == native_id))
            )
        overlap[classifier_id] = counts

    native_overlaps: dict[int, set[int]] = {
        native_id: set() for native_id in range(1, native_count + 1)
    }
    for classifier_id, counts in overlap.items():
        for native_id, count in counts.items():
            if count > 0:
                native_overlaps[native_id].add(classifier_id)

    reciprocal: dict[int, int] = {}
    for native_id in range(1, native_count + 1):
        candidates = native_overlaps[native_id]
        if len(candidates) != 1:
            continue
        classifier_id = next(iter(candidates))
        if set(overlap.get(classifier_id, {})) == {native_id}:
            # A score is transferable only for a strict physical-overlap
            # bijection: one native component to one classifier component in
            # both directions. Never select a largest-overlap fallback.
            reciprocal[native_id] = classifier_id

    reserved_display_ids = set(scores)
    used_display_ids: set[int] = set()
    next_fallback = 1
    rois: list[LesionROI] = []
    for native_id in range(1, native_count + 1):
        classifier_id = reciprocal.get(native_id)
        pi_rads = scores.get(classifier_id) if classifier_id is not None else None
        unambiguous = (
            classifier_id is not None
            and classifier_id in scores
        )
        if unambiguous:
            display_id = int(classifier_id)
            match_status = "matched_by_reciprocal_physical_overlap"
        else:
            while next_fallback in used_display_ids or next_fallback in reserved_display_ids:
                next_fallback += 1
            display_id = next_fallback
            next_fallback += 1
            pi_rads = None
            match_status = "ambiguous_score_unavailable"
        used_display_ids.add(display_id)
        rois.append(
            LesionROI(
                display_id=display_id,
                native_component_id=native_id,
                classifier_component_id=classifier_id,
                pi_rads=pi_rads,
                match_status=match_status,
                mask=native_labels == native_id,
            )
        )
    rois.sort(key=lambda item: item.display_id)

    mapping = {
        "native_component_count": int(native_count),
        "classifier_component_count": int(len(classifier_ids)),
        "reported_score_count": int(len(scores)),
        "overlap_voxels": {
            str(classifier_id): {
                str(native_id): count for native_id, count in sorted(counts.items())
            }
            for classifier_id, counts in sorted(overlap.items())
        },
        "lesions": [
            {
                "display_id": roi.display_id,
                "native_component_id": roi.native_component_id,
                "classifier_component_id": roi.classifier_component_id,
                "pi_rads": roi.pi_rads,
                "match_status": roi.match_status,
                "roi_name": roi.name,
                "native_voxels": int(np.count_nonzero(roi.mask)),
            }
            for roi in rois
        ],
    }
    return rois, mapping


def _prepare_rtutils_series(series: SourceSeries, parent: Path) -> Path:
    directory = parent / "t2_for_rtutils"
    directory.mkdir()
    for index, item in enumerate(series.slices):
        destination = directory / f"slice_{index:04d}.dcm"
        source_header = item.header
        if "StudyID" in source_header:
            shutil.copy2(item.path, destination)
            continue
        dataset = pydicom.dcmread(item.path, force=True)
        # StudyID is Type 2. A zero-length value is truthful and lets rt-utils
        # build the derived object without inventing an accession-based value.
        dataset.StudyID = ""
        pydicom.dcmwrite(destination, dataset, enforce_file_format=True)
    return directory


def _polygon_area_mm2(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    area_vector = 0.5 * np.sum(
        np.cross(points, np.roll(points, -1, axis=0)),
        axis=0,
    )
    return float(np.linalg.norm(area_vector))


def _format_decimal_string(value: float) -> str:
    """Format one DICOM DS value without exceeding its 16-character VR."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ReviewExportError("Cannot encode a non-finite DICOM decimal string")
    for precision in range(12, 5, -1):
        encoded = f"{numeric:.{precision}g}"
        if len(encoded) <= DICOM_TEXT_VR_MAX_LENGTH["DS"]:
            return encoded
    raise ReviewExportError(f"Cannot encode DICOM decimal string: {numeric!r}")


def _repair_contour(
    contour: Dataset,
    pixel_axis_column: np.ndarray,
    pixel_axis_row: np.ndarray,
    *,
    require_single_polygon: bool = False,
) -> list[Dataset]:
    points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ReviewExportError("RTSTRUCT contour contains non-finite coordinates")
    if len(points) > 1 and np.allclose(points[0], points[-1], rtol=0.0, atol=1e-8):
        points = points[:-1]
    unique = np.unique(np.round(points, decimals=8), axis=0)
    if len(unique) >= 3 and _polygon_area_mm2(points) > 1e-6:
        repaired = copy.deepcopy(contour)
        repaired.ContourData = [
            _format_decimal_string(value) for value in points.reshape(-1)
        ]
        repaired.NumberOfContourPoints = len(points)
        return [repaired]
    if not len(unique):
        raise ReviewExportError("RTSTRUCT contour contains no coordinates")

    if require_single_polygon:
        corners = np.concatenate(
            [
                np.asarray(
                    [
                        center - 0.49 * pixel_axis_column - 0.49 * pixel_axis_row,
                        center + 0.49 * pixel_axis_column - 0.49 * pixel_axis_row,
                        center + 0.49 * pixel_axis_column + 0.49 * pixel_axis_row,
                        center - 0.49 * pixel_axis_column + 0.49 * pixel_axis_row,
                    ]
                )
                for center in unique
            ],
            axis=0,
        )
        unit_column = pixel_axis_column / np.linalg.norm(pixel_axis_column)
        unit_row = pixel_axis_row / np.linalg.norm(pixel_axis_row)
        projected = np.column_stack(
            [corners @ unit_column, corners @ unit_row]
        )
        hull = ConvexHull(projected)
        polygon = corners[hull.vertices]
        repaired = copy.deepcopy(contour)
        repaired.ContourData = [
            _format_decimal_string(value) for value in polygon.reshape(-1)
        ]
        repaired.NumberOfContourPoints = len(polygon)
        return [repaired]

    repaired_items: list[Dataset] = []
    for center in unique:
        axis_column = 0.49 * pixel_axis_column
        axis_row = 0.49 * pixel_axis_row
        square = np.asarray(
            [
                center - axis_column - axis_row,
                center + axis_column - axis_row,
                center + axis_column + axis_row,
                center - axis_column + axis_row,
            ],
            dtype=float,
        )
        repaired = copy.deepcopy(contour)
        repaired.ContourData = [
            _format_decimal_string(value) for value in square.reshape(-1)
        ]
        repaired.NumberOfContourPoints = len(square)
        repaired_items.append(repaired)
    return repaired_items


def _copy_or_clear(dataset: Dataset, source: Dataset, keyword: str) -> None:
    if keyword in source:
        dataset.add(copy.deepcopy(source.data_element(keyword)))
    elif keyword in dataset:
        del dataset[keyword]


def _finalize_rtstruct(
    path: Path,
    series: SourceSeries,
    *,
    roi_metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    dataset = pydicom.dcmread(path)
    source = series.first
    now = dt.datetime.now(dt.timezone.utc)

    for keyword in PATIENT_STUDY_KEYWORDS:
        _copy_or_clear(dataset, source, keyword)
    source_metadata_normalizations: list[dict[str, Any]] = []
    source_study_id = str(getattr(source, "StudyID", "") or "")
    source_accession = str(getattr(source, "AccessionNumber", "") or "")
    for keyword, value in (
        ("StudyID", source_study_id),
        ("AccessionNumber", source_accession),
    ):
        if len(value) <= 16:
            setattr(dataset, keyword, value)
        else:
            # Both attributes use VR SH. Never truncate an identifier into a
            # different value: emit the legal Type-2 empty value and preserve
            # auditable evidence of the source defect without duplicating PHI.
            setattr(dataset, keyword, "")
            source_metadata_normalizations.append(
                {
                    "keyword": keyword,
                    "action": "set_zero_length_due_to_source_vr_violation",
                    "source_length": len(value),
                    "source_value_sha256": hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest(),
                    "maximum_length": 16,
                }
            )
    for keyword in SOURCE_TYPE2_KEYWORDS:
        if keyword not in dataset:
            setattr(dataset, keyword, "")
            source_metadata_normalizations.append(
                {
                    "keyword": keyword,
                    "action": "insert_zero_length_for_missing_type_2_attribute",
                }
            )
    dataset.StudyInstanceUID = series.study_instance_uid
    dataset.FrameOfReferenceUID = series.frame_of_reference_uid
    dataset.PositionReferenceIndicator = str(
        getattr(source, "PositionReferenceIndicator", "") or ""
    )
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = generate_uid()
    dataset.file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = RTStructureSetStorage
    dataset.Modality = "RTSTRUCT"
    dataset.SeriesNumber = ""
    dataset.OperatorsName = ""
    dataset.InstanceNumber = 1
    dataset.InstanceCreationDate = now.strftime("%Y%m%d")
    dataset.InstanceCreationTime = now.strftime("%H%M%S.%f")
    dataset.SeriesDate = now.strftime("%Y%m%d")
    dataset.SeriesTime = now.strftime("%H%M%S.%f")
    dataset.StructureSetDate = now.strftime("%Y%m%d")
    dataset.StructureSetTime = now.strftime("%H%M%S.%f")
    dataset.StructureSetLabel = "AI_PROSTATE"
    dataset.StructureSetName = "AI Prostate and Lesions"
    dataset.StructureSetDescription = (
        "Patient-specific model segmentation converted from NIfTI"
    )
    dataset.SeriesDescription = "AI Prostate and Lesions"
    dataset.ApprovalStatus = "UNAPPROVED"
    dataset.Manufacturer = ""
    for keyword in (
        "ManufacturerModelName",
        "SoftwareVersions",
        "DeviceSerialNumber",
        "InstitutionName",
        "InstitutionAddress",
        "StationName",
    ):
        if keyword in dataset:
            del dataset[keyword]
    dataset.remove_private_tags()

    for item in getattr(dataset, "StructureSetROISequence", []):
        number = int(item.ROINumber)
        metadata = roi_metadata[number]
        item.ROIName = metadata["name"]
        item.ROIDescription = metadata["description"]
        item.ROIGenerationAlgorithm = "AUTOMATIC"
        item.ROIGenerationDescription = metadata["generation_description"]

    for item in getattr(dataset, "RTROIObservationsSequence", []):
        number = int(item.ReferencedROINumber)
        metadata = roi_metadata[number]
        item.ROIObservationLabel = metadata["observation_label"][:16]
        item.RTROIInterpretedType = metadata["interpreted_type"]
        item.ROIInterpreter = ""
        if "ROIObservationDescription" in item:
            del item.ROIObservationDescription
        item.remove_private_tags()

    orientation = np.asarray(source.ImageOrientationPatient, dtype=float)
    row_spacing, column_spacing = (float(value) for value in source.PixelSpacing)
    pixel_axis_column = orientation[:3] * column_spacing
    pixel_axis_row = orientation[3:] * row_spacing
    repaired_count = 0
    contour_count = 0
    source_sops = set(series.sop_instance_uids)
    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        roi_number = int(roi_contour.ReferencedROINumber)
        is_prostate = roi_metadata[roi_number]["name"] == "Prostate"
        finalized: list[Dataset] = []
        for contour in getattr(roi_contour, "ContourSequence", []):
            original_points = np.asarray(
                contour.ContourData,
                dtype=float,
            ).reshape(-1, 3)
            if len(original_points) > 1 and np.allclose(
                original_points[0],
                original_points[-1],
                rtol=0.0,
                atol=1e-8,
            ):
                original_points = original_points[:-1]
            if (
                len(np.unique(np.round(original_points, 8), axis=0)) < 3
                or _polygon_area_mm2(original_points) <= 1e-6
            ):
                repaired_count += 1
            repaired = _repair_contour(
                contour,
                pixel_axis_column,
                pixel_axis_row,
                require_single_polygon=is_prostate,
            )
            finalized.extend(repaired)
        roi_contour.ContourSequence = finalized
        for contour_number, contour in enumerate(finalized, start=1):
            contour.ContourNumber = contour_number
            contour.ContourGeometricType = "CLOSED_PLANAR"
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            contour.ContourData = [
                _format_decimal_string(value) for value in points.reshape(-1)
            ]
            contour.NumberOfContourPoints = len(points)
            if len(points) < 3 or len(np.unique(np.round(points, 8), axis=0)) < 3:
                raise ReviewExportError("Unable to create a non-degenerate contour")
            if _polygon_area_mm2(points) <= 1e-6:
                raise ReviewExportError("Unable to create a positive-area contour")
            if np.max(np.abs((points - points[0]) @ series.normal)) > 1e-3:
                raise ReviewExportError("RTSTRUCT contour is not coplanar")
            references = getattr(contour, "ContourImageSequence", [])
            if len(references) != 1:
                raise ReviewExportError(
                    "Every contour must reference exactly one T2 slice"
                )
            if str(references[0].ReferencedSOPInstanceUID) not in source_sops:
                raise ReviewExportError("Contour references a non-source T2 object")
            contour_count += 1

    pydicom.dcmwrite(
        path,
        dataset,
        enforce_file_format=True,
        little_endian=True,
        implicit_vr=False,
    )
    return {
        "contour_count": contour_count,
        "repaired_degenerate_contour_count": repaired_count,
        "series_instance_uid": str(dataset.SeriesInstanceUID),
        "sop_instance_uid": str(dataset.SOPInstanceUID),
        "source_metadata_normalizations": source_metadata_normalizations,
    }


def _referenced_t2_sops(dataset: Dataset) -> set[str]:
    referenced: set[str] = set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                for image in getattr(series, "ContourImageSequence", []):
                    referenced.add(str(image.ReferencedSOPInstanceUID))
    return referenced


def _validate_bounded_text_vrs(dataset: Dataset) -> list[str]:
    errors: list[str] = []

    def visit(_parent: Dataset, element) -> None:
        maximum = DICOM_TEXT_VR_MAX_LENGTH.get(element.VR)
        if maximum is None:
            return
        values = list(element.value) if element.VM > 1 else [element.value]
        for value in values:
            if value is None:
                continue
            text = str(value)
            representations = text.split("=") if element.VR == "PN" else [text]
            if any(len(item) > maximum for item in representations):
                keyword = element.keyword or str(element.tag)
                errors.append(
                    f"{element.VR} value exceeds {maximum} characters:{keyword}"
                )

    dataset.walk(visit)
    return errors


def _referenced_series_uids(dataset: Dataset) -> set[str]:
    referenced: set[str] = set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            for series in getattr(study, "RTReferencedSeriesSequence", []):
                referenced.add(str(series.SeriesInstanceUID))
    return referenced


def _referenced_study_uids(dataset: Dataset) -> set[str]:
    referenced: set[str] = set()
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", []):
        for study in getattr(frame, "RTReferencedStudySequence", []):
            referenced.add(str(study.ReferencedSOPInstanceUID))
    return referenced


def _run_external_dicom_validator(path: Path) -> dict[str, Any]:
    executable = shutil.which("dciodvfy")
    if executable is None:
        return {
            "status": "unavailable",
            "command": "dciodvfy",
            "return_code": None,
            "output": "",
        }
    result = subprocess.run(
        [executable, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "command": executable,
        "return_code": result.returncode,
        "output": output[-12000:],
    }


def _validate_rtstruct(
    path: Path,
    prepared_series_directory: Path,
    series: SourceSeries,
    expected_rois: dict[str, np.ndarray],
    roi_metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    dataset = pydicom.dcmread(path)
    errors: list[str] = []
    if str(dataset.SOPClassUID) != str(RTStructureSetStorage):
        errors.append("wrong_sop_class")
    if str(dataset.file_meta.TransferSyntaxUID) != str(ExplicitVRLittleEndian):
        errors.append("not_explicit_vr_little_endian")
    if str(dataset.file_meta.MediaStorageSOPInstanceUID) != str(
        dataset.SOPInstanceUID
    ):
        errors.append("file_meta_sop_instance_uid_mismatch")
    if str(dataset.StudyInstanceUID) != series.study_instance_uid:
        errors.append("study_uid_mismatch")
    if str(getattr(dataset, "FrameOfReferenceUID", "")) != (
        series.frame_of_reference_uid
    ):
        errors.append("top_level_frame_of_reference_mismatch")
    if str(getattr(dataset, "PositionReferenceIndicator", "") or "") != str(
        getattr(series.first, "PositionReferenceIndicator", "") or ""
    ):
        errors.append("position_reference_indicator_mismatch")
    if str(dataset.SeriesInstanceUID) == series.series_instance_uid:
        errors.append("derived_series_uid_reuses_source")
    if str(dataset.SOPInstanceUID) in set(series.sop_instance_uids):
        errors.append("derived_sop_uid_reuses_source")
    if not UID(str(dataset.SeriesInstanceUID)).is_valid:
        errors.append("invalid_derived_series_uid")
    if not UID(str(dataset.SOPInstanceUID)).is_valid:
        errors.append("invalid_derived_sop_uid")
    frame_uids = {
        str(item.FrameOfReferenceUID)
        for item in getattr(dataset, "ReferencedFrameOfReferenceSequence", [])
    }
    if frame_uids != {series.frame_of_reference_uid}:
        errors.append("frame_of_reference_mismatch")
    if _referenced_study_uids(dataset) != {series.study_instance_uid}:
        errors.append("source_study_reference_mismatch")
    if _referenced_series_uids(dataset) != {series.series_instance_uid}:
        errors.append("source_t2_series_reference_mismatch")
    if _referenced_t2_sops(dataset) != set(series.sop_instance_uids):
        errors.append("source_t2_reference_set_mismatch")
    if str(getattr(dataset, "StructureSetLabel", "")) != "AI_PROSTATE":
        errors.append("unexpected_structure_set_label")
    if str(getattr(dataset, "ApprovalStatus", "")) != "UNAPPROVED":
        errors.append("approval_status_not_unapproved")
    if "Manufacturer" not in dataset:
        errors.append("manufacturer_missing")
    elif str(dataset.Manufacturer):
        errors.append("manufacturer_not_empty")
    for keyword in (
        "ManufacturerModelName",
        "SoftwareVersions",
        "DeviceSerialNumber",
    ):
        if keyword in dataset:
            errors.append(f"unexpected_{keyword}")
    for keyword in REQUIRED_TYPE2_KEYWORDS:
        if keyword not in dataset:
            errors.append(f"missing_type_2_attribute:{keyword}")
    for keyword in PATIENT_STUDY_KEYWORDS:
        source_has_value = keyword in series.first
        derived_has_value = keyword in dataset
        if not source_has_value:
            if keyword in SOURCE_TYPE2_KEYWORDS:
                if not derived_has_value or str(dataset.get(keyword, "")):
                    errors.append(f"missing_source_type_2_not_preserved:{keyword}")
            elif derived_has_value:
                errors.append(f"invented_optional_source_attribute:{keyword}")
            continue
        if not derived_has_value:
            errors.append(f"missing_copied_source_attribute:{keyword}")
            continue
        if keyword == "AccessionNumber" and len(
            str(series.first.AccessionNumber or "")
        ) > DICOM_TEXT_VR_MAX_LENGTH["SH"]:
            if str(dataset.AccessionNumber):
                errors.append("invalid_source_accession_not_cleared")
            continue
        if dataset.data_element(keyword).value != series.first.data_element(
            keyword
        ).value:
            errors.append(f"copied_source_attribute_mismatch:{keyword}")
    source_study_id = str(getattr(series.first, "StudyID", "") or "")
    expected_study_id = (
        source_study_id
        if len(source_study_id) <= DICOM_TEXT_VR_MAX_LENGTH["SH"]
        else ""
    )
    if str(getattr(dataset, "StudyID", "") or "") != expected_study_id:
        errors.append("copied_source_attribute_mismatch:StudyID")
    errors.extend(_validate_bounded_text_vrs(dataset))

    names_by_number = {
        int(item.ROINumber): str(item.ROIName)
        for item in getattr(dataset, "StructureSetROISequence", [])
    }
    expected_names_by_number = {
        number: str(metadata["name"])
        for number, metadata in roi_metadata.items()
    }
    if names_by_number != expected_names_by_number:
        errors.append("roi_name_set_mismatch")
    if set(names_by_number.values()) != set(expected_rois):
        errors.append("roi_mask_name_set_mismatch")
    for item in getattr(dataset, "StructureSetROISequence", []):
        number = int(item.ROINumber)
        if str(getattr(item, "ROIGenerationAlgorithm", "")) != "AUTOMATIC":
            errors.append(f"roi_not_automatic:{number}")
        if str(getattr(item, "ReferencedFrameOfReferenceUID", "")) != (
            series.frame_of_reference_uid
        ):
            errors.append(f"roi_frame_of_reference_mismatch:{number}")
    observations = {
        int(item.ReferencedROINumber): item
        for item in getattr(dataset, "RTROIObservationsSequence", [])
    }
    for number, name in names_by_number.items():
        observation = observations.get(number)
        expected_type = "ORGAN" if name == "Prostate" else "GTV"
        if observation is None:
            errors.append(f"missing_observation:{name}")
            continue
        if str(getattr(observation, "RTROIInterpretedType", "")) != expected_type:
            errors.append(f"wrong_interpreted_type:{name}")
        metadata = roi_metadata[number]
        if str(getattr(observation, "ROIObservationLabel", "")) != str(
            metadata["observation_label"]
        ):
            errors.append(f"wrong_observation_label:{name}")
        if "ROIObservationDescription" in observation:
            errors.append(f"packed_observation_description_present:{name}")
    if set(observations) != set(names_by_number):
        errors.append("roi_observation_number_set_mismatch")

    contour_count = 0
    minimum_points: int | None = None
    contour_numbers: set[int] = set()
    prostate_contours_by_sop: dict[str, int] = {}
    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        roi_number = int(roi_contour.ReferencedROINumber)
        contour_numbers.add(roi_number)
        expected_color = [int(value) for value in roi_metadata[roi_number]["color"]]
        actual_color = [int(value) for value in roi_contour.ROIDisplayColor]
        if actual_color != expected_color:
            errors.append(f"wrong_roi_color:{roi_number}")
        for contour in getattr(roi_contour, "ContourSequence", []):
            contour_count += 1
            if str(contour.ContourGeometricType) != "CLOSED_PLANAR":
                errors.append("non_closed_planar_contour")
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            minimum_points = len(points) if minimum_points is None else min(minimum_points, len(points))
            if len(points) < 3 or len(np.unique(np.round(points, 8), axis=0)) < 3:
                errors.append("degenerate_contour")
            if len(points) > 1 and np.allclose(points[0], points[-1], atol=1e-8, rtol=0.0):
                errors.append("repeated_closing_point")
            if np.max(np.abs((points - points[0]) @ series.normal)) > 1e-3:
                errors.append("non_planar_contour")
            refs = getattr(contour, "ContourImageSequence", [])
            if (
                len(refs) != 1
                or str(refs[0].ReferencedSOPClassUID) != str(MRImageStorage)
                or str(refs[0].ReferencedSOPInstanceUID)
                not in set(series.sop_instance_uids)
            ):
                errors.append("invalid_contour_image_reference")
            elif names_by_number.get(roi_number) == "Prostate":
                sop_uid = str(refs[0].ReferencedSOPInstanceUID)
                prostate_contours_by_sop[sop_uid] = (
                    prostate_contours_by_sop.get(sop_uid, 0) + 1
                )
    if contour_numbers != set(names_by_number):
        errors.append("roi_contour_number_set_mismatch")
    expected_prostate_sops = {
        series.slices[int(index)].header.SOPInstanceUID
        for index in np.flatnonzero(
            np.any(expected_rois["Prostate"], axis=(0, 1))
        )
    }
    if set(prostate_contours_by_sop) != {
        str(value) for value in expected_prostate_sops
    }:
        errors.append("prostate_contour_slice_set_mismatch")
    if any(count != 1 for count in prostate_contours_by_sop.values()):
        errors.append("prostate_has_multiple_contours_on_a_slice")

    round_trip_differences: dict[str, int] = {}
    try:
        loaded = RTStructBuilder.create_from(
            dicom_series_path=str(prepared_series_directory),
            rt_struct_path=str(path),
        )
        for name, expected in expected_rois.items():
            actual = loaded.get_roi_mask_by_name(name)
            expected_rtutils = np.transpose(expected.astype(bool), (1, 0, 2))
            difference = int(np.count_nonzero(actual ^ expected_rtutils))
            round_trip_differences[name] = difference
            if difference:
                errors.append(f"round_trip_mismatch:{name}:{difference}")
    except Exception as exc:
        errors.append(f"round_trip_error:{type(exc).__name__}:{exc}")

    external = _run_external_dicom_validator(path)
    if external["status"] == "failed":
        errors.append("external_dicom_validator_failed")
    return {
        "valid": not errors,
        "errors": errors,
        "contour_count": contour_count,
        "minimum_contour_points": minimum_points,
        "round_trip_different_voxels": round_trip_differences,
        "external_validator": external,
    }


def _create_rtstruct(
    output_path: Path,
    prepared_series_directory: Path,
    series: SourceSeries,
    prostate_mask: np.ndarray,
    lesions: Sequence[LesionROI],
) -> dict[str, Any]:
    rtstruct = RTStructBuilder.create_new(
        dicom_series_path=str(prepared_series_directory)
    )
    prostate_rtutils = np.transpose(prostate_mask.astype(bool), (1, 0, 2))
    rtstruct.add_roi(
        mask=prostate_rtutils,
        color=[0, 255, 0],
        name="Prostate",
        description="Patient-specific model prostate segmentation",
        use_pin_hole=False,
        approximate_contours=False,
        roi_generation_algorithm="AUTOMATIC",
    )
    expected_rois: dict[str, np.ndarray] = {"Prostate": prostate_mask.astype(bool)}
    for index, lesion in enumerate(lesions):
        color = list(LESION_COLORS[index % len(LESION_COLORS)])
        rtstruct.add_roi(
            mask=np.transpose(lesion.mask.astype(bool), (1, 0, 2)),
            color=color,
            name=lesion.name,
            description="Patient-specific model lesion segmentation",
            use_pin_hole=False,
            approximate_contours=False,
            roi_generation_algorithm="AUTOMATIC",
        )
        expected_rois[lesion.name] = lesion.mask.astype(bool)
    rtstruct.save(str(output_path))

    roi_metadata: dict[int, dict[str, Any]] = {}
    dataset = pydicom.dcmread(output_path, stop_before_pixels=True)
    for roi in dataset.StructureSetROISequence:
        number = int(roi.ROINumber)
        name = str(roi.ROIName)
        if name == "Prostate":
            roi_metadata[number] = {
                "name": name,
                "description": "Patient-specific model prostate segmentation",
                "generation_description": (
                    "Largest component and filled holes per occupied T2 slice"
                ),
                "observation_label": "PROSTATE",
                "interpreted_type": "ORGAN",
                "color": [0, 255, 0],
            }
        else:
            lesion = next(item for item in lesions if item.name == name)
            roi_metadata[number] = {
                "name": name,
                "description": "Patient-specific model lesion segmentation",
                "generation_description": "Unchanged native lesion component",
                "observation_label": lesion.observation_label,
                "interpreted_type": "GTV",
                "color": list(
                    LESION_COLORS[
                        next(
                            index
                            for index, item in enumerate(lesions)
                            if item.name == name
                        )
                        % len(LESION_COLORS)
                    ]
                ),
            }

    finalization = _finalize_rtstruct(
        output_path,
        series,
        roi_metadata=roi_metadata,
    )
    validation = _validate_rtstruct(
        output_path,
        prepared_series_directory,
        series,
        expected_rois,
        roi_metadata,
    )
    if not validation["valid"]:
        raise ReviewExportError(
            "Generated canonical RTSTRUCT failed validation: "
            + "; ".join(validation["errors"])
        )
    return {**finalization, "validation": validation}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def generate_accepted_rtstruct(
    case_directory: Path,
    *,
    validate_only: bool = False,
    output_name: str = CANONICAL_RTSTRUCT_NAME,
    config: RTStructConfig = RTStructConfig(),
) -> dict[str, Any]:
    """Create the accepted, non-smoothed combined RTSTRUCT profile.

    The prostate is derived from ``organ/cleaned_organ.nii.gz`` using the
    radiologist-accepted single-outer-contour representation. Lesions are
    unchanged native 26-connected components from ``lesion/lesion_mask.nii.gz``
    and are matched to ``lesions.txt`` scores only by reciprocal physical
    overlap. ROI numbering is deterministic: prostate is 1 and lesions are
    2..N. Generation and validation occur in temporary storage; publication is
    one atomic replacement after every structural and round-trip check passes.

    No prostate smoothing is performed or published by this function.
    """
    case_directory = Path(case_directory).resolve()
    _validate_export_config(config)
    missing = case_missing_prerequisites(case_directory)
    if missing:
        raise IneligibleCaseError(
            f"Case {case_directory.name} is missing prerequisites: "
            + ", ".join(missing)
        )
    if Path(output_name).name != output_name:
        raise ValueError("output_name must be a filename in the case root")

    organ_path = case_directory / "organ/cleaned_organ.nii.gz"
    lesion_path = case_directory / "lesion/lesion_mask.nii.gz"
    organ_image = nib.load(str(organ_path))
    lesion_image = nib.load(str(lesion_path))
    series = _read_source_series(
        case_directory / "dicom/t2",
        grid_tolerance_mm=config.grid_tolerance_mm,
    )
    _validate_nifti_grid(
        organ_image,
        series,
        affine_tolerance_mm=config.affine_tolerance_mm,
    )
    _validate_nifti_grid(
        lesion_image,
        series,
        affine_tolerance_mm=config.affine_tolerance_mm,
    )
    _validate_matching_nifti_grids(
        organ_image,
        lesion_image,
        affine_tolerance_mm=config.affine_tolerance_mm,
    )

    organ_mask = _binary_mask(organ_image)
    lesion_mask = _binary_mask(lesion_image)
    scores = _parse_lesion_scores(case_directory / "lesions.txt")
    lesion_rois, lesion_mapping = build_lesion_rois(
        lesion_mask,
        organ_image.affine,
        scores,
    )
    prostate_profile = build_single_outer_prostate_profile(
        organ_mask,
        lesion_mask,
        organ_image.affine,
        config,
    )
    if not prostate_profile.accepted:
        raise ReviewExportError(
            "Accepted single-outer prostate profile failed: "
            + "; ".join(prostate_profile.failed_gates)
        )

    with tempfile.TemporaryDirectory(
        prefix=".accepted_rtstruct_",
        dir=case_directory,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        prepared_series = _prepare_rtutils_series(series, staging)
        staged_output = staging / output_name
        result = _create_rtstruct(
            staged_output,
            prepared_series,
            series,
            prostate_profile.mask,
            lesion_rois,
        )

        dataset = pydicom.dcmread(staged_output, stop_before_pixels=True)
        expected_numbers = list(range(1, len(lesion_rois) + 2))
        sequence_numbers = {
            "structure_set": [
                int(item.ROINumber) for item in dataset.StructureSetROISequence
            ],
            "contours": [
                int(item.ReferencedROINumber)
                for item in dataset.ROIContourSequence
            ],
            "observations": [
                int(item.ReferencedROINumber)
                for item in dataset.RTROIObservationsSequence
            ],
        }
        if any(values != expected_numbers for values in sequence_numbers.values()):
            raise ReviewExportError(
                "Canonical ROI numbering is not prostate 1 followed by lesions 2..N: "
                f"{sequence_numbers}"
            )

        result.update(
            {
                "published": not validate_only,
                "output_name": output_name,
                "prostate_profile": prostate_profile.metrics,
                "lesion_mapping": lesion_mapping,
                "roi_numbers": sequence_numbers,
            }
        )
        if not validate_only:
            os.replace(staged_output, case_directory / output_name)
    return _json_safe(result)
