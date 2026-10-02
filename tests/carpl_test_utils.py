"""Shared fixtures for the CARPL integration tests (synthetic data only)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ComprehensiveSRStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    PYDICOM_IMPLEMENTATION_UID,
    RTStructureSetStorage,
    generate_uid,
)

ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "carpl" / "common"
SCRIPTS_DIR = ROOT / "scripts"

os.environ.setdefault("CARPL_LIB_DIR", str(COMMON_DIR))
for _path in (str(COMMON_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def load_module(name: str, relative_path: str):
    """Import a CARPL service module from its path (folders contain hyphens)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _file_meta(sop_class: str, sop_instance: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_instance
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    return meta


def make_mr(
    study_uid: str,
    series_uid: str,
    *,
    description: str = "t2_tse_tra",
    instance_number: int = 1,
    modality: str = "MR",
    patient_id: str = "SYNTHETIC-001",
    sop_uid: "str | None" = None,
) -> Dataset:
    ds = Dataset()
    ds.file_meta = _file_meta(MRImageStorage, sop_uid or generate_uid())
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = generate_uid()
    ds.PatientID = patient_id
    ds.PatientName = "Synthetic^Test"
    ds.Modality = modality
    ds.SeriesDescription = description
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.Rows = ds.Columns = 2
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = b"\x00\x01" * 4
    return ds


def encode(ds: Dataset) -> bytes:
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, enforce_file_format=True)
    return buffer.getvalue()


def write(ds: Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return path


def write_combined_rtstruct(path: Path, study_uid: str, patient_id: str) -> Path:
    ds = Dataset()
    ds.file_meta = _file_meta(RTStructureSetStorage, generate_uid())
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = patient_id
    ds.Modality = "RTSTRUCT"
    ds.StructureSetLabel = "AI_PROSTATE"
    ds.ApprovalStatus = "UNAPPROVED"
    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "Prostate"
    roi.ROIGenerationAlgorithm = "AUTOMATIC"
    ds.StructureSetROISequence = [roi]
    polygon = Dataset()
    polygon.ContourGeometricType = "CLOSED_PLANAR"
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = [polygon]
    ds.ROIContourSequence = [roi_contour]
    observation = Dataset()
    observation.ReferencedROINumber = 1
    observation.RTROIInterpretedType = "ORGAN"
    ds.RTROIObservationsSequence = [observation]
    return write(ds, path)


def write_sr(path: Path, study_uid: str, patient_id: str) -> Path:
    ds = Dataset()
    ds.file_meta = _file_meta(ComprehensiveSRStorage, generate_uid())
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = patient_id
    ds.Modality = "SR"
    return write(ds, path)


def create_complete_output(output: Path, study_uid: str, t2_series_uid: str, patient_id: str = "SYNTHETIC-001") -> None:
    """A synthetic output folder that case_status classifies as SUCCESS_COMPLETE."""
    import case_status

    for relative in case_status.EXPECTED_FILES.values():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "cleanup_metrics.json":
            path.write_text(json.dumps({"original_foreground_voxels": 100, "removed_voxels": 0,
                                        "filled_hole_voxels": 0}), encoding="utf-8")
        else:
            path.touch()
    for relative in case_status.LESION_FOLD_FILES:
        (output / relative).touch()
    write_combined_rtstruct(output / "combined_organ_lesion_RTSTRUCT.dcm", study_uid, patient_id)
    write_sr(output / "prostate_measurements_SR.dcm", study_uid, patient_id)
    for role, description in (("t2", "t2_tse_tra"), ("adc", "ep2d_diff_tra_ADC"), ("highb", "ep2d_diff_tra")):
        series_uid = t2_series_uid if role == "t2" else generate_uid()
        write(make_mr(study_uid, series_uid, description=description, patient_id=patient_id),
              output / "dicom" / role / "1.dcm")
