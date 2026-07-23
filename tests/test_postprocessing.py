"""Guardrail tests for prostate mask cleanup and physical measurements."""

import sys
import unittest
from pathlib import Path

import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "prostate_mri_lesion_seg_app"
sys.path.insert(0, str(APP_DIR))

from common import clean_organ_mask, compute_prostate_measurements  # noqa: E402


class OrganCleanupTests(unittest.TestCase):

    def test_removes_islands_and_preserves_multiclass_labels(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[3:12, 3:8, 3:12] = 1
        mask[3:12, 8:12, 3:12] = 2
        mask[16:18, 16:18, 16:18] = 2

        cleaned, metrics = clean_organ_mask(
            mask,
            fill_holes=False,
            return_metrics=True,
        )

        self.assertTrue(np.all(cleaned[3:12, 3:8, 3:12] == 1))
        self.assertTrue(np.all(cleaned[3:12, 8:12, 3:12] == 2))
        self.assertEqual(int(cleaned[16:18, 16:18, 16:18].sum()), 0)
        self.assertEqual(metrics["original_component_count"], 2)
        self.assertEqual(metrics["removed_island_count"], 1)
        self.assertEqual(metrics["removed_voxels"], 8)

    def test_filled_holes_receive_nearest_class_label(self):
        mask = np.ones((9, 9, 9), dtype=np.uint8)
        mask[:, 5:, :] = 2
        mask[4, 4, 4] = 0

        cleaned, metrics = clean_organ_mask(
            mask,
            fill_holes=True,
            return_metrics=True,
        )

        self.assertNotEqual(int(cleaned[4, 4, 4]), 0)
        self.assertIn(int(cleaned[4, 4, 4]), (1, 2))
        self.assertEqual(metrics["filled_hole_voxels"], 1)

    def test_full_connectivity_preserves_diagonal_connection(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1, 1, 1] = 1
        mask[2, 2, 2] = 1

        cleaned, metrics = clean_organ_mask(
            mask,
            fill_holes=False,
            return_metrics=True,
        )

        self.assertEqual(int(cleaned.sum()), 2)
        self.assertEqual(metrics["original_component_count"], 1)

    def test_empty_mask_is_safe(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        cleaned, metrics = clean_organ_mask(
            mask,
            return_metrics=True,
        )
        self.assertTrue(np.array_equal(cleaned, mask))
        self.assertEqual(metrics["original_foreground_voxels"], 0)


class ProstateMeasurementTests(unittest.TestCase):

    def test_affine_volume_and_ras_dimensions(self):
        mask = np.ones((10, 20, 30), dtype=np.uint8)
        affine = np.diag([0.5, 1.0, 2.0, 1.0])

        result = compute_prostate_measurements(mask, affine=affine)

        self.assertEqual(result["left_right_mm"], 5.0)
        self.assertEqual(result["anterior_posterior_mm"], 20.0)
        self.assertEqual(result["superior_inferior_mm"], 60.0)
        self.assertEqual(result["volume_mm3"], 6000.0)
        self.assertEqual(result["volume_cc"], 6.0)

    def test_axis_permutation_maps_to_patient_axes(self):
        mask = np.ones((4, 5, 6), dtype=np.uint8)
        affine = np.array([
            [0.0, -2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        result = compute_prostate_measurements(mask, affine=affine)

        self.assertEqual(result["left_right_mm"], 10.0)
        self.assertEqual(result["anterior_posterior_mm"], 4.0)
        self.assertEqual(result["superior_inferior_mm"], 18.0)
        self.assertEqual(result["volume_mm3"], 720.0)


if __name__ == "__main__":
    unittest.main()
