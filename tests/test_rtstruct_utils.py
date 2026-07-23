"""Guardrails for standards-based no-keyhole RTSTRUCT generation."""

import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    PYDICOM_IMPLEMENTATION_UID,
    generate_uid,
)
from rt_utils import RTStructBuilder


APP_DIR = Path(__file__).resolve().parents[1] / "prostate_mri_lesion_seg_app"
sys.path.insert(0, str(APP_DIR))

from rtstruct_utils import create_rtstruct_from_mask  # noqa: E402


def _create_mr_series(directory: Path, slice_count: int = 3) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    for index in range(slice_count):
        path = directory / f"slice_{index:03d}.dcm"
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        dataset = FileDataset(
            str(path),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        dataset.SOPClassUID = MRImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.Modality = "MR"
        dataset.PatientID = "TEST001"
        dataset.PatientName = "TEST^PATIENT"
        dataset.PatientBirthDate = ""
        dataset.PatientSex = ""
        dataset.StudyDate = "20260723"
        dataset.StudyTime = "120000"
        dataset.StudyID = "TESTSTUDY"
        dataset.SeriesNumber = 1
        dataset.InstanceNumber = index + 1
        dataset.Rows = 8
        dataset.Columns = 8
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelSpacing = [0.7, 1.3]
        dataset.SliceThickness = 1.0
        dataset.ImagePositionPatient = [0.0, 0.0, float(index)]
        dataset.ImageOrientationPatient = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        dataset.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        dataset.save_as(path, enforce_file_format=True)


class RTStructContourTests(unittest.TestCase):

    def test_xor_export_has_no_keyholes_or_degenerate_contours(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _create_mr_series(root)
            output_path = root / "test_RTSTRUCT.dcm"

            mask = np.zeros((8, 8, 3), dtype=bool)
            mask[2, 2, 0] = True
            mask[4, 2:4, 1] = True
            mask[3, 4, 1] = True
            mask[2, 4, 1] = True
            mask[1:7, 1:7, 2] = True
            mask[3:5, 3:5, 2] = False

            nifti_path = root / "mask.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.transpose(mask, (1, 0, 2)).astype(np.uint8),
                    np.eye(4),
                ),
                nifti_path,
            )
            create_rtstruct_from_mask(
                dicom_series_path=str(root),
                nifti_mask_path=str(nifti_path),
                output_rtstruct_path=str(output_path),
                roi_name="Prostate",
                roi_color=[0, 255, 0],
            )

            dataset = pydicom.dcmread(output_path)
            contours = dataset.ROIContourSequence[0].ContourSequence
            self.assertTrue(contours)
            for contour in contours:
                self.assertEqual(
                    contour.ContourGeometricType,
                    "CLOSEDPLANAR_XOR",
                )
                points = np.asarray(
                    contour.ContourData,
                    dtype=np.float64,
                ).reshape(-1, 3)
                self.assertGreaterEqual(len(points), 3)
                self.assertGreaterEqual(
                    len(np.unique(np.round(points, decimals=8), axis=0)),
                    3,
                )

            loaded = RTStructBuilder.create_from(
                dicom_series_path=str(root),
                rt_struct_path=str(output_path),
            )
            round_trip_mask = loaded.get_roi_mask_by_name("Prostate")
            np.testing.assert_array_equal(round_trip_mask, mask)

    def test_explicit_pinhole_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Pinhole"):
            create_rtstruct_from_mask(
                dicom_series_path="/unused",
                nifti_mask_path="/unused",
                output_rtstruct_path="/unused",
                use_pin_hole=True,
            )


if __name__ == "__main__":
    unittest.main()
