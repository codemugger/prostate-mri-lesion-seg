from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from training.cohort import (
    AUDIT_COLUMNS,
    MANIFEST_COLUMNS,
    audit_eligibility,
    clean_binary_organ_mask,
    create_splits,
    write_csv,
)
from training.datasets import select_records
from training.engine import (
    _continued_run_dir,
    _enforce_requested_gpu_visibility,
    _scheduler,
    _validate_config_contract,
    build_model,
)
from training.export import load_weights, package_deployment_bundle
from training.inference_tiler import inference_ranges, lesion_tiled_predict
from training.losses import ProbabilityDiceFocalLoss
from training.metrics import lesion_detection_counts
from training.transforms import (
    get_organ_train_transforms,
    get_organ_val_transforms,
)


REPOSITORY = Path(__file__).resolve().parents[1]


class CohortPolicyTests(unittest.TestCase):
    def _audit_row(self, reviewed="Y", q1="Y", q2="Y", q3="Y", q4="Y"):
        return {
            AUDIT_COLUMNS["reviewed"]: reviewed,
            AUDIT_COLUMNS["q1"]: q1,
            AUDIT_COLUMNS["q2"]: q2,
            AUDIT_COLUMNS["q3"]: q3,
            AUDIT_COLUMNS["q4"]: q4,
        }

    def test_exact_audit_rules(self):
        self.assertEqual(audit_eligibility(self._audit_row()), (True, True))
        self.assertEqual(
            audit_eligibility(self._audit_row(q1="N")), (False, True)
        )
        self.assertEqual(
            audit_eligibility(self._audit_row(q2="N")), (True, False)
        )
        self.assertEqual(
            audit_eligibility(self._audit_row(reviewed="P")), (False, False)
        )
        self.assertEqual(
            audit_eligibility(self._audit_row(reviewed="")), (False, False)
        )

    def test_cleanup_removes_islands_and_fills_holes(self):
        mask = np.zeros((9, 9, 9), dtype=np.uint8)
        mask[2:7, 2:7, 2:7] = 2
        mask[4, 4, 4] = 0
        mask[0, 0, 0] = 1
        cleaned, metrics = clean_binary_organ_mask(mask)
        self.assertEqual(set(np.unique(cleaned)), {0, 1})
        self.assertEqual(cleaned[0, 0, 0], 0)
        self.assertEqual(cleaned[4, 4, 4], 1)
        self.assertEqual(metrics["removed_island_count"], 1)
        self.assertEqual(metrics["filled_hole_voxels"], 1)

    def test_splits_are_deterministic_and_disjoint(self):
        rows = []
        for index in range(80):
            rows.append(
                {
                    "source": "SGH" if index < 60 else "ProstateX",
                    "subject_id": f"case-{index:03d}",
                    "organ_eligible": "true",
                    "lesion_eligible": "true",
                    "manufacturer": "SIEMENS" if index % 2 else "GE",
                    "study_year": str(2015 + index % 8),
                    "lesion_is_empty": "true" if index % 5 == 0 else "false",
                }
            )
        first = create_splits(
            rows,
            task="lesion",
            seed=42,
            test_fraction=0.15,
            validation_fraction=0.15,
        )
        second = create_splits(
            rows,
            task="lesion",
            seed=42,
            test_fraction=0.15,
            validation_fraction=0.15,
        )
        self.assertEqual(first, second)
        test = {
            (row["source"], row["subject_id"])
            for row in first
            if row["split"] == "test"
        }
        development = {
            (row["source"], row["subject_id"])
            for row in first
            if row["split"] == "development"
        }
        self.assertFalse(test & development)
        self.assertEqual(len(test | development), len(rows))
        self.assertTrue({row["fold"] for row in first if row["fold"]} <= set("01234"))


class TrainingContractTests(unittest.TestCase):
    def test_organ_affine_augmentation_preserves_exact_patch_shape(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.random.default_rng(42).normal(
                size=(132, 131, 18)
            ).astype(np.float32)
            label = np.zeros_like(image, dtype=np.uint8)
            label[35:95, 35:95, 3:15] = 1
            image_path = root / "image.nii.gz"
            label_path = root / "label.nii.gz"
            nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
            nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)

            transform = get_organ_train_transforms(
                spacing=(1.0, 1.0, 1.0),
                roi_size=(128, 128, 16),
                samples_per_image=2,
                pos_to_neg=1.0,
            )
            for _ in range(20):
                samples = transform(
                    {"image": str(image_path), "label": str(label_path)}
                )
                self.assertEqual(len(samples), 2)
                for sample in samples:
                    self.assertEqual(
                        tuple(sample["image"].shape), (1, 128, 128, 16)
                    )
                    self.assertEqual(
                        tuple(sample["label"].shape), (1, 128, 128, 16)
                    )
                    np.testing.assert_allclose(
                        sample["image"].affine,
                        sample["label"].affine,
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_organ_validation_resamples_label_to_exact_image_grid(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((64, 65, 9), dtype=np.float32)
            label = np.zeros((80, 78, 9), dtype=np.uint8)
            label[20:60, 18:58, 2:7] = 1
            image_affine = np.diag([0.8, 0.8, 3.0, 1.0])
            label_affine = np.diag([0.64, 2.0 / 3.0, 3.0, 1.0])
            label_affine[:3, 3] = (0.25, -0.4, 0.0)
            image_path = root / "image.nii.gz"
            label_path = root / "label.nii.gz"
            nib.save(nib.Nifti1Image(image, image_affine), image_path)
            nib.save(nib.Nifti1Image(label, label_affine), label_path)

            sample = get_organ_val_transforms((1.0, 1.0, 1.0))(
                {"image": str(image_path), "label": str(label_path)}
            )
            self.assertEqual(sample["image"].shape, sample["label"].shape)
            np.testing.assert_allclose(
                sample["image"].affine,
                sample["label"].affine,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_hive_single_gpu_guard_fails_closed(self):
        with mock.patch.dict(os.environ, {"HIVE_REQUIRE_SINGLE_GPU": "1"}):
            with mock.patch(
                "training.engine.torch.cuda.device_count", return_value=2
            ):
                with self.assertRaisesRegex(RuntimeError, "PyTorch can see 2"):
                    _enforce_requested_gpu_visibility()

    def test_resume_continues_original_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            checkpoints = run / "checkpoints"
            checkpoints.mkdir(parents=True)
            (run / "resolved_config.yaml").write_text("train: {}\n")
            (run / "cohort_provenance.json").write_text("{}\n")
            last = run / "model_last.pth.tar"
            snapshot = checkpoints / "model_epoch_0050.pth.tar"
            last.write_bytes(b"last")
            snapshot.write_bytes(b"snapshot")
            self.assertEqual(_continued_run_dir(last), run)
            self.assertEqual(_continued_run_dir(snapshot), run)

    def test_default_configs_retain_periodic_snapshots(self):
        expected_intervals = {"organ": 50, "lesion": 10}
        for name, expected_interval in expected_intervals.items():
            config = yaml.safe_load((REPOSITORY / f"configs/{name}.yaml").read_text())
            self.assertEqual(
                config["output"]["snapshot_interval"], expected_interval
            )
            self.assertEqual(
                config["scheduler"]["t_max"], config["train"]["epochs"]
            )
            _validate_config_contract(config, task=name)

    def test_scheduler_rejects_accidental_cosine_cycles(self):
        config = yaml.safe_load((REPOSITORY / "configs/lesion.yaml").read_text())
        config["scheduler"]["t_max"] = config["train"]["epochs"] - 1
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with self.assertRaisesRegex(ValueError, "must equal train.epochs"):
            _scheduler(config, optimizer)

    def test_lesion_hive_plan_is_sequential_and_conservative(self):
        config = yaml.safe_load((REPOSITORY / "configs/lesion.yaml").read_text())
        self.assertEqual(config["train"]["epochs"], 50)
        self.assertEqual(config["scheduler"]["t_max"], 50)
        self.assertEqual(config["train"]["lr"], 1e-4)
        self.assertEqual(config["train"]["grad_accum_steps"], 2)
        self.assertEqual(config["train"]["val_interval"], 1)
        self.assertEqual(config["output"]["save_last_interval"], 1)
        self.assertEqual(config["output"]["snapshot_interval"], 10)

        notebook = json.loads(
            (REPOSITORY / "hive_train_lesion.ipynb").read_text()
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("START_ALL_FOLDS = False", code)
        self.assertIn("FOLDS_TO_TRAIN = list(range(5))", code)
        self.assertIn(
            "for position, fold in enumerate(FOLDS_TO_TRAIN", code
        )
        self.assertNotIn("ThreadPoolExecutor", code)

    def test_manifest_join_never_uses_held_out_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_rows = []
            split_rows = []
            for index in range(7):
                row = {column: "" for column in MANIFEST_COLUMNS}
                row.update(
                    {
                        "source": "SGH",
                        "subject_id": f"case-{index}",
                        "organ_eligible": "true",
                        "lesion_eligible": "true",
                    }
                )
                manifest_rows.append(row)
                split_rows.append(
                    {
                        "source": "SGH",
                        "subject_id": f"case-{index}",
                        "split": "test" if index == 6 else "development",
                        "fold": "" if index == 6 else str(index % 5),
                        "stratum": "",
                    }
                )
            manifest = root / "manifest.csv"
            splits = root / "splits.csv"
            write_csv(manifest, manifest_rows, MANIFEST_COLUMNS)
            write_csv(
                splits,
                split_rows,
                ["source", "subject_id", "split", "fold", "stratum"],
            )
            train = select_records(
                manifest,
                splits,
                task="lesion",
                partition="train",
                fold=0,
            )
            validation = select_records(
                manifest,
                splits,
                task="lesion",
                partition="val",
                fold=0,
            )
            self.assertFalse(
                {row["subject_id"] for row in train}
                & {row["subject_id"] for row in validation}
            )
            self.assertNotIn("case-6", {row["subject_id"] for row in train})
            self.assertNotIn("case-6", {row["subject_id"] for row in validation})

    def test_probability_loss_is_finite_and_backpropagates(self):
        logits = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
        probabilities = torch.softmax(logits, dim=1)
        labels = torch.randint(0, 2, (2, 1, 8, 8, 8))
        loss = ProbabilityDiceFocalLoss()(probabilities, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_inference_tiler_covers_tail_ranges(self):
        class Network(torch.nn.Module):
            def forward(self, value):
                foreground = torch.sigmoid(value[:, :1])
                return torch.cat([1.0 - foreground, foreground], dim=1)

        self.assertEqual(inference_ranges(65), [(0, 64), (1, 65)])
        image = torch.randn(1, 3, 33, 65, 32)
        output = lesion_tiled_predict(Network(), image)
        self.assertEqual(tuple(output.shape), (1, 2, 33, 65, 32))
        torch.testing.assert_close(output.sum(dim=1), torch.ones_like(output[:, 0]))

    def test_multifocal_lesions_are_not_reduced_to_largest_component(self):
        target = np.zeros((20, 20, 20), dtype=bool)
        target[2:4, 2:4, 2:4] = True
        target[12:15, 12:15, 12:15] = True
        prediction = target.copy()
        prediction[6:8, 15:17, 3:5] = True
        self.assertEqual(lesion_detection_counts(prediction, target), (2, 0, 1))

    def test_current_deployment_weights_strict_load(self):
        organ_cfg = yaml.safe_load((REPOSITORY / "configs/organ.yaml").read_text())
        lesion_cfg = yaml.safe_load((REPOSITORY / "configs/lesion.yaml").read_text())
        organ_path = REPOSITORY / "models/organ/model.ts"
        lesion_path = REPOSITORY / "models/fold0/model_best_fold0.pth.tar"
        if not organ_path.is_file() or not lesion_path.is_file():
            self.skipTest("Deployment weights are not present")
        organ = build_model(organ_cfg)
        load_weights(organ, organ_path, map_location="cpu")
        lesion = build_model(lesion_cfg)
        load_weights(lesion, lesion_path, map_location="cpu")
        self.assertEqual(organ(torch.randn(1, 1, 32, 32, 16)).shape[1], 2)

    def test_packaging_uses_exact_inference_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index in range(7):
                source = root / f"source-{index}"
                source.write_bytes(f"weights-{index}".encode())
                sources.append(source)
            destination = root / "bundle"
            package_deployment_bundle(
                destination=destination,
                organ_model=sources[0],
                lesion_models=sources[1:6],
                classifier_model=sources[6],
            )
            self.assertTrue((destination / "organ/model.ts").is_file())
            for fold in range(5):
                self.assertTrue(
                    (
                        destination
                        / f"fold{fold}/model_best_fold{fold}.pth.tar"
                    ).is_file()
                )
            self.assertTrue(
                (destination / "classifier/model_best.pth.tar").is_file()
            )
            self.assertTrue((destination / "deployment_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
