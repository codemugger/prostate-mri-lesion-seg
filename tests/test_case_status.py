"""Audit guardrails for the current pipeline output contract."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


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
)


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

    def test_complete_current_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._create_complete_case(root)

            status = inspect_case(root, patient_id="TEST")

            self.assertEqual(status.status, Status.SUCCESS_COMPLETE)
            self.assertEqual(status.cleanup_metrics["status"], "CLEANED")
            self.assertTrue(status.file_flags["prostate_sr"])

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
