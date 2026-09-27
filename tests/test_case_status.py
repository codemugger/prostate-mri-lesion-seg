"""Audit guardrails for the current pipeline output contract."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID,
    RTStructureSetStorage,
    generate_uid,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from case_status import (  # noqa: E402
    CSV_COLUMNS,
    CaseStatus,
    EXPECTED_FILES,
    LESION_FOLD_FILES,
    Status,
    append_csv_row,
    inspect_case,
    is_current_combined_rtstruct,
)


def _write_current_combined_rtstruct(path: Path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = RTStructureSetStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StructureSetLabel = "AI_PROSTATE"
    dataset.ApprovalStatus = "UNAPPROVED"

    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "Prostate"
    roi.ROIGenerationAlgorithm = "AUTOMATIC"
    dataset.StructureSetROISequence = [roi]

    polygon = Dataset()
    polygon.ContourGeometricType = "CLOSED_PLANAR"
    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = [polygon]
    dataset.ROIContourSequence = [roi_contour]

    observation = Dataset()
    observation.ReferencedROINumber = 1
    observation.RTROIInterpretedType = "ORGAN"
    dataset.RTROIObservationsSequence = [observation]
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)


class CaseStatusTests(unittest.TestCase):

    def _create_complete_case(self, root: Path) -> None:
        for relative_path in EXPECTED_FILES.values():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "cleanup_metrics.json":
                path.write_text(
                    json.dumps({
                        "original_foreground_voxels": 100,
                        "original_component_count": 2,
                        "removed_island_count": 1,
                        "removed_voxels": 5,
                        "filled_hole_voxels": 2,
                        "largest_component_fraction": 0.95,
                    }),
                    encoding="utf-8",
                )
            else:
                path.touch()
        for relative_path in LESION_FOLD_FILES:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        _write_current_combined_rtstruct(
            root / EXPECTED_FILES["combined_rtstruct"]
        )

    def test_complete_current_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_complete_case(root)

            status = inspect_case(root, patient_id="TEST")

            self.assertEqual(status.status, Status.SUCCESS_COMPLETE)
            self.assertEqual(status.cleanup_metrics["status"], "CLEANED")
            self.assertTrue(status.file_flags["prostate_sr"])

    def test_legacy_combined_rtstruct_is_not_current(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_complete_case(root)
            rtstruct_path = root / EXPECTED_FILES["combined_rtstruct"]
            dataset = pydicom.dcmread(rtstruct_path)
            dataset.StructureSetLabel = "RTSTRUCT"
            pydicom.dcmwrite(rtstruct_path, dataset, enforce_file_format=True)

            self.assertFalse(is_current_combined_rtstruct(rtstruct_path))
            status = inspect_case(root, patient_id="TEST")
            self.assertEqual(status.status, Status.FAIL_RTSTRUCT)

    def test_empty_canonical_rtstruct_requires_proven_empty_organ(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_complete_case(root)
            rtstruct_path = root / EXPECTED_FILES["combined_rtstruct"]
            dataset = pydicom.dcmread(rtstruct_path)
            dataset.StructureSetROISequence = []
            dataset.ROIContourSequence = []
            dataset.RTROIObservationsSequence = []
            pydicom.dcmwrite(rtstruct_path, dataset, enforce_file_format=True)

            self.assertFalse(is_current_combined_rtstruct(rtstruct_path))

            metrics_path = root / EXPECTED_FILES["cleanup_metrics"]
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["original_foreground_voxels"] = 0
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            for relative_path in LESION_FOLD_FILES:
                (root / relative_path).unlink()

            self.assertTrue(is_current_combined_rtstruct(rtstruct_path))
            status = inspect_case(root, patient_id="TEST")
            self.assertEqual(status.status, Status.SUCCESS_EMPTY_LESION)

    def test_missing_sr_is_reporting_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_complete_case(root)
            (root / EXPECTED_FILES["prostate_sr"]).unlink()

            status = inspect_case(root, patient_id="TEST")

            self.assertEqual(status.status, Status.FAIL_REPORTING)

    def test_missing_cleaned_mask_is_postprocessing_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_complete_case(root)
            (root / EXPECTED_FILES["cleaned_organ_nii"]).unlink()

            status = inspect_case(root, patient_id="TEST")

            self.assertEqual(status.status, Status.FAIL_POSTPROCESSING)

    def test_old_audit_header_is_migrated_before_append(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "audit.csv"
            csv_path.write_text(
                "patient_id,status\nOLD,SUCCESS_COMPLETE\n",
                encoding="utf-8",
            )
            append_csv_row(
                csv_path,
                CaseStatus(
                    patient_id="NEW",
                    status=Status.SUCCESS_COMPLETE,
                    reason="complete",
                ),
            )

            lines = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(","), CSV_COLUMNS)
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
