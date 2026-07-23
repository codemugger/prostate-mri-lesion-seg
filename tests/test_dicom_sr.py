"""Round-trip validation for the prostate DICOM Structured Report."""

import sys
import tempfile
import unittest
from pathlib import Path

import highdicom as hd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    PYDICOM_IMPLEMENTATION_UID,
    RawDataStorage,
    generate_uid,
)


APP_DIR = Path(__file__).resolve().parents[1] / "prostate_mri_lesion_seg_app"
sys.path.insert(0, str(APP_DIR))

from dicom_sr_utils import generate_prostate_measurement_sr  # noqa: E402


def _create_source_mr(path: Path) -> FileDataset:
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
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.Modality = "MR"
    dataset.PatientID = "TEST001"
    dataset.PatientName = "TEST^PATIENT"
    dataset.PatientBirthDate = ""
    dataset.PatientSex = ""
    dataset.StudyDate = "20260723"
    dataset.StudyTime = "120000"
    dataset.StudyID = "TESTSTUDY"
    dataset.AccessionNumber = "TESTACCESSION"
    dataset.ReferringPhysicianName = ""
    dataset.SeriesNumber = 1
    dataset.InstanceNumber = 1
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.save_as(path, enforce_file_format=True)
    return dataset


class ProstateStructuredReportTests(unittest.TestCase):

    def test_round_trip_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.dcm"
            source = _create_source_mr(source_path)
            raw_data = _create_source_mr(root / "raw_data.dcm")
            raw_data.SOPClassUID = RawDataStorage
            raw_data.file_meta.MediaStorageSOPClassUID = RawDataStorage
            raw_data.save_as(root / "raw_data.dcm", enforce_file_format=True)
            output_path = root / "prostate_measurements_SR.dcm"

            generate_prostate_measurement_sr(
                dicom_series_path=root,
                output_path=output_path,
                measurements={
                    "left_right_mm": 42.0,
                    "anterior_posterior_mm": 35.0,
                    "superior_inferior_mm": 51.0,
                    "volume_mm3": 39000.0,
                    "volume_cc": 39.0,
                },
                cleanup_metrics={
                    "original_component_count": 3,
                    "removed_island_count": 2,
                },
            )

            report = pydicom.dcmread(output_path)
            self.assertEqual(report.Modality, "SR")
            self.assertEqual(report.CompletionFlag, "COMPLETE")
            self.assertEqual(report.VerificationFlag, "UNVERIFIED")
            self.assertEqual(report.StudyInstanceUID, source.StudyInstanceUID)
            self.assertEqual(report.PatientID, source.PatientID)
            hd.sr.ComprehensiveSR.from_dataset(report)

            code_meanings = []
            numeric_values = []
            def collect_content_items(dataset, data_element):
                if data_element.keyword == "ConceptNameCodeSequence":
                    code_meanings.append(
                        data_element.value[0].CodeMeaning
                    )
                elif data_element.keyword == "MeasuredValueSequence":
                    numeric_values.append(
                        float(data_element.value[0].NumericValue)
                    )
            report.walk(collect_content_items)
            self.assertIn("Prostate left-right extent", code_meanings)
            self.assertIn("Prostate anterior-posterior extent", code_meanings)
            self.assertIn("Prostate superior-inferior extent", code_meanings)
            self.assertIn(39000.0, numeric_values)
            self.assertIn(39.0, numeric_values)


if __name__ == "__main__":
    unittest.main()
