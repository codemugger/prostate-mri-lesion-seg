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


PROSTATE_MEASUREMENTS = {
    "left_right_mm": 42.0,
    "anterior_posterior_mm": 35.0,
    "superior_inferior_mm": 51.0,
    "volume_mm3": 39000.0,
    "volume_cc": 39.0,
}


def _write_case(root: Path, *, organ: bool, lesion: bool, ambiguous_score: bool = False) -> None:
    """Synthetic case outputs: cleaned organ, lesion mask, probability map, lesions.txt."""
    import nibabel as nib
    import numpy as np
    import yaml

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    shape = (24, 24, 12)
    organ_mask = np.zeros(shape, dtype=np.uint8)
    lesion_mask = np.zeros(shape, dtype=np.uint8)
    probability = np.zeros(shape, dtype=np.float64)
    if organ:
        organ_mask[4:20, 4:20, 2:10] = 1
    if lesion:
        lesion_mask[8:12, 8:12, 4:7] = 1
        probability[8:12, 8:12, 4:7] = 0.7
        probability[9, 9, 5] = 0.912345
    (root / "organ").mkdir(parents=True, exist_ok=True)
    (root / "lesion").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(organ_mask, affine), str(root / "organ/cleaned_organ.nii.gz"))
    nib.save(nib.Nifti1Image(lesion_mask, affine), str(root / "lesion/lesion_mask.nii.gz"))
    if lesion:
        nib.save(nib.Nifti1Image(probability, affine), str(root / "lesion/merged_lesion_prob.nii.gz"))
    rows = [{"Organ": "Prostate", "Volume_mm3": 39000.0, "Volume_cc": 39.0}]
    if lesion:
        rows.append({
            "Lesion_ID": 2 if ambiguous_score else 1,
            "Major_Axis_Length": 7.123456,
            "Volume": 48.0,
            "PI_RADS": 4,
        })
    (root / "lesions.txt").write_text(yaml.dump(rows), encoding="utf-8")


def _lesion_groups(report) -> "list[dict[str, object]]":
    """Return the TID 1501 groups whose finding type is 'Lesion'."""
    groups = []
    for item in report.ContentSequence:
        if item.ConceptNameCodeSequence[0].CodeMeaning != "Imaging Measurements":
            continue
        for group in item.ContentSequence:
            content: "dict[str, object]" = {}
            for child in group.ContentSequence:
                name = child.ConceptNameCodeSequence[0].CodeMeaning
                if child.ValueType == "NUM":
                    measured = child.MeasuredValueSequence[0]
                    content[name] = (
                        float(measured.NumericValue),
                        measured.MeasurementUnitsCodeSequence[0].CodeValue,
                    )
                elif child.ValueType == "CODE":
                    content[name] = child.ConceptCodeSequence[0].CodeMeaning
                elif child.ValueType == "TEXT":
                    content[name] = child.TextValue
            if content.get("Finding") == "Lesion":
                groups.append(content)
    return groups


class LesionFindingsReportTests(unittest.TestCase):
    """CARPL DICOM Push Method 2 content: finding, probability, references."""

    def _generate(self, **case_options):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        source = _create_source_mr(root / "source.dcm")
        _write_case(root, **case_options)
        output_path = root / "prostate_measurements_SR.dcm"
        generate_prostate_measurement_sr(
            dicom_series_path=root,
            output_path=output_path,
            measurements=PROSTATE_MEASUREMENTS,
        )
        report = pydicom.dcmread(output_path)
        hd.sr.ComprehensiveSR.from_dataset(report)
        self.assertEqual(report.StudyInstanceUID, source.StudyInstanceUID)
        self.assertEqual(report.VerificationFlag, "UNVERIFIED")
        return report

    def test_detected_lesion_has_finding_probability_and_pi_rads(self):
        report = self._generate(organ=True, lesion=True)
        groups = _lesion_groups(report)
        self.assertEqual(len(groups), 1)
        lesion = groups[0]
        self.assertEqual(lesion["Tracking Identifier"], "Lesion 1")
        self.assertEqual(lesion["Finding Site"], "Prostate")
        self.assertEqual(lesion["Probability"], (91.23, "%"))
        self.assertEqual(lesion["AI PI-RADS assessment category"], "PI-RADS 4")
        self.assertEqual(lesion["AI lesion volume"], (48.0, "mm3"))
        self.assertEqual(lesion["AI lesion major axis length"], (7.12, "mm"))
        self.assertEqual(report.ContentDescription, "Prostate measurements and AI lesion findings")
        # The observer context identifies the AI device (CARPL requirement).
        observer_names = []
        report.walk(lambda ds, el: observer_names.append(ds.TextValue)
                    if el.keyword == "TextValue" and "Prostate MRI" in str(el.value) else None)
        self.assertIn("Prostate MRI Lesion Segmentation", observer_names)

    def test_unmatched_lesion_score_is_reported_unavailable(self):
        report = self._generate(organ=True, lesion=True, ambiguous_score=True)
        lesion = _lesion_groups(report)[0]
        self.assertEqual(lesion["AI PI-RADS assessment category"], "PI-RADS unavailable")
        self.assertEqual(lesion["Probability"], (91.23, "%"))
        self.assertNotIn("AI lesion volume", lesion)

    def test_prostate_without_lesion_reports_absent(self):
        report = self._generate(organ=True, lesion=False)
        groups = _lesion_groups(report)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["Lesion"], "Absent")
        self.assertNotIn("Probability", groups[0])

    def test_no_prostate_makes_no_lesion_claim(self):
        report = self._generate(organ=False, lesion=False)
        self.assertEqual(_lesion_groups(report), [])
        self.assertEqual(report.ContentDescription, "Cleaned prostate dimensions and volume")

    def test_incomplete_lesion_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _create_source_mr(root / "source.dcm")
            _write_case(root, organ=True, lesion=True)
            (root / "lesion/lesion_mask.nii.gz").unlink()
            with self.assertRaises(FileNotFoundError):
                generate_prostate_measurement_sr(
                    dicom_series_path=root,
                    output_path=root / "prostate_measurements_SR.dcm",
                    measurements=PROSTATE_MEASUREMENTS,
                )


if __name__ == "__main__":
    unittest.main()
