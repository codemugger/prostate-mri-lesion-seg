"""Regression guardrails for the accepted UroFusion RTSTRUCT profile."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    MRImageStorage,
    PYDICOM_IMPLEMENTATION_UID,
    RTStructureSetStorage,
    generate_uid,
)
import yaml


APP_DIR = Path(__file__).resolve().parents[1] / "prostate_mri_lesion_seg_app"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from urofusion_rtstruct import (  # noqa: E402
    GeometryMismatchError,
    RTStructConfig,
    build_lesion_rois,
    build_single_outer_prostate_profile,
    generate_accepted_rtstruct,
)
from rtstruct_utils import generate_rtstruct_files  # noqa: E402
from validate_rtstruct_outputs import validate_case  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mr_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    frame_uid: str,
    index: int,
    slice_count: int,
) -> None:
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.Modality = "MR"
    dataset.PatientName = "SYNTHETIC^PATIENT"
    dataset.PatientID = "SYNTHETIC_CASE"
    dataset.PatientBirthDate = ""
    dataset.PatientSex = "O"
    dataset.StudyDate = "20260927"
    dataset.StudyTime = "120000"
    dataset.ReferringPhysicianName = ""
    dataset.AccessionNumber = "ACCESSION1"
    dataset.StudyID = "STUDY1"
    dataset.SeriesNumber = 7
    dataset.InstanceNumber = index + 1
    dataset.Rows = 48
    dataset.Columns = 48
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.SpacingBetweenSlices = 1.0
    dataset.ImagePositionPatient = [0.0, 0.0, float(index)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelData = np.full((48, 48), index, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _create_case(case: Path, lesion_count: int) -> dict[str, object]:
    t2 = case / "dicom" / "t2"
    organ_dir = case / "organ"
    lesion_dir = case / "lesion"
    t2.mkdir(parents=True)
    organ_dir.mkdir(parents=True)
    lesion_dir.mkdir(parents=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    slice_count = 17
    for index in range(slice_count):
        _write_mr_slice(
            t2 / f"slice_{index:03d}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            frame_uid=frame_uid,
            index=index,
            slice_count=slice_count,
        )

    shape = (48, 48, slice_count)
    coordinates = np.indices(shape, dtype=float)
    center = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    radii = np.asarray([14.0, 12.0, 7.0])
    organ = sum(
        ((coordinates[axis] - center[axis]) / radii[axis]) ** 2
        for axis in range(3)
    ) <= 1.0
    lesion = np.zeros(shape, dtype=bool)
    if lesion_count >= 1:
        lesion[18:20, 23:25, 8] = True
    if lesion_count >= 2:
        lesion[28:30, 23:25, 8] = True
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    nib.save(
        nib.Nifti1Image(organ.astype(np.uint16), affine),
        organ_dir / "cleaned_organ.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(organ.astype(np.uint16), affine),
        organ_dir / "organ.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(lesion.astype(np.uint8), affine),
        lesion_dir / "lesion_mask.nii.gz",
    )
    (case / "lesions.txt").write_text(
        yaml.safe_dump(
            [
                {"Lesion_ID": index, "PI_RADS": index + 3}
                for index in range(1, lesion_count + 1)
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {"study_uid": study_uid, "series_uid": series_uid, "frame_uid": frame_uid}


class ProstateProfileTests(unittest.TestCase):
    def test_single_outer_profile_is_deterministic_and_never_changes_lesions(self):
        source = np.zeros((20, 20, 3), dtype=bool)
        source[4:16, 4:16, 1] = True
        source[9:11, 9:11, 1] = False
        source[1:3, 1:3, 1] = True
        lesion = np.zeros_like(source)
        lesion[1, 1, 1] = True
        source_before = source.copy()
        lesion_before = lesion.copy()

        first = build_single_outer_prostate_profile(source, lesion, np.eye(4))
        second = build_single_outer_prostate_profile(source, lesion, np.eye(4))

        self.assertTrue(first.accepted)
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(lesion, lesion_before)
        self.assertEqual(first.metrics["removed_component_voxels"], 4)
        self.assertEqual(first.metrics["filled_hole_voxels"], 4)

    def test_single_outer_profile_keeps_eight_percent_limit(self):
        source = np.zeros((20, 20, 3), dtype=bool)
        source[4:16, 4:16, 1] = True
        source[7:13, 7:13, 1] = False
        result = build_single_outer_prostate_profile(
            source,
            np.zeros_like(source),
            np.eye(4),
            RTStructConfig(max_volume_delta_percent=8.0),
        )
        self.assertFalse(result.accepted)
        self.assertIn(
            "single_contour_profile_volume_delta_exceeds_limit",
            result.failed_gates,
        )


class LesionMappingTests(unittest.TestCase):
    def test_zero_one_and_multiple_native_lesions(self):
        mask = np.zeros((20, 20, 10), dtype=bool)
        rois, mapping = build_lesion_rois(mask, np.eye(4), {})
        self.assertEqual(rois, [])
        self.assertEqual(mapping["native_component_count"], 0)
        mask[4:7, 4:7, 3:6] = True
        rois, _ = build_lesion_rois(mask, np.eye(4), {1: 4})
        self.assertEqual([item.name for item in rois], ["Lesion 1 - AI PI-RADS 4"])
        mask[13:16, 13:16, 3:6] = True
        rois, _ = build_lesion_rois(mask, np.eye(4), {1: 4, 2: 5})
        self.assertEqual(
            [item.name for item in rois],
            ["Lesion 1 - AI PI-RADS 4", "Lesion 2 - AI PI-RADS 5"],
        )

    def test_ambiguous_score_mapping_never_uses_nearest_centroid(self):
        native = np.zeros((9, 9, 5), dtype=bool)
        native[1, 4, 2] = True
        native[7, 4, 2] = True
        classifier = np.zeros_like(native, dtype=np.int32)
        classifier[1:8, 4, 2] = 1
        with mock.patch(
            "urofusion_rtstruct._classifier_grid_labels",
            return_value=(classifier, np.eye(4)),
        ):
            rois, _ = build_lesion_rois(native, np.eye(4), {1: 5})
        self.assertTrue(all(item.pi_rads is None for item in rois))
        self.assertTrue(all("unavailable" in item.name for item in rois))


class CanonicalRTStructTests(unittest.TestCase):
    def test_independent_four_file_validator_accepts_mixed_profiles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case = Path(temporary_directory) / "case"
            _create_case(case, 2)
            generate_rtstruct_files(
                output_folder=case,
                input_folder=case,
                t2_dicom_series_path=case / "dicom" / "t2",
            )
            generate_accepted_rtstruct(case)

            result = validate_case(case)

            self.assertEqual(result["valid"], "true", result["errors"])
            self.assertEqual(result["rtstruct_count"], "4")
            self.assertEqual(result["all_expected_contour_geometry"], "true")
            self.assertEqual(result["roundtrip_exact"], "true")

    def test_one_and_two_lesion_cases_share_one_flat_numbering_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for lesion_count in (1, 2):
                with self.subTest(lesion_count=lesion_count):
                    case = root / f"case_{lesion_count}"
                    fixture = _create_case(case, lesion_count)
                    source_paths = [
                        case / "organ/cleaned_organ.nii.gz",
                        case / "lesion/lesion_mask.nii.gz",
                        case / "lesions.txt",
                    ]
                    before = [_sha256(path) for path in source_paths]

                    validated = generate_accepted_rtstruct(case, validate_only=True)
                    self.assertFalse(validated["published"])
                    self.assertFalse(
                        (case / "combined_organ_lesion_RTSTRUCT.dcm").exists()
                    )
                    result = generate_accepted_rtstruct(case)

                    self.assertTrue(result["published"])
                    self.assertTrue(result["validation"]["valid"])
                    expected_numbers = list(range(1, lesion_count + 2))
                    self.assertEqual(
                        result["roi_numbers"],
                        {
                            "structure_set": expected_numbers,
                            "contours": expected_numbers,
                            "observations": expected_numbers,
                        },
                    )
                    dataset = pydicom.dcmread(
                        case / "combined_organ_lesion_RTSTRUCT.dcm"
                    )
                    self.assertEqual(dataset.SOPClassUID, RTStructureSetStorage)
                    self.assertEqual(
                        dataset.file_meta.TransferSyntaxUID,
                        ExplicitVRLittleEndian,
                    )
                    self.assertEqual(dataset.StructureSetLabel, "AI_PROSTATE")
                    self.assertEqual(dataset.ApprovalStatus, "UNAPPROVED")
                    self.assertEqual(dataset.StudyInstanceUID, fixture["study_uid"])
                    self.assertEqual(dataset.FrameOfReferenceUID, fixture["frame_uid"])
                    self.assertNotEqual(dataset.SeriesInstanceUID, fixture["series_uid"])
                    self.assertEqual(
                        [str(item.ROIName) for item in dataset.StructureSetROISequence],
                        ["Prostate"]
                        + [
                            f"Lesion {index} - AI PI-RADS {index + 3}"
                            for index in range(1, lesion_count + 1)
                        ],
                    )
                    self.assertEqual(
                        [
                            str(item.ROIObservationLabel)
                            for item in dataset.RTROIObservationsSequence
                        ],
                        ["PROSTATE"]
                        + [f"LESION_{index}" for index in range(1, lesion_count + 1)],
                    )
                    self.assertTrue(
                        all(
                            contour.ContourGeometricType == "CLOSED_PLANAR"
                            for roi in dataset.ROIContourSequence
                            for contour in roi.ContourSequence
                        )
                    )
                    self.assertEqual(before, [_sha256(path) for path in source_paths])
                    self.assertFalse(any(case.glob("*smooth*")))

    def test_grid_mismatch_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case = Path(temporary_directory) / "case"
            _create_case(case, 1)
            organ_path = case / "organ/cleaned_organ.nii.gz"
            image = nib.load(organ_path)
            affine = image.affine.copy()
            affine[0, 3] += 2.0
            nib.save(
                nib.Nifti1Image(
                    np.asarray(image.dataobj),
                    affine,
                    header=image.header.copy(),
                ),
                organ_path,
            )
            with self.assertRaisesRegex(GeometryMismatchError, "does not match"):
                generate_accepted_rtstruct(case)
            self.assertFalse(
                (case / "combined_organ_lesion_RTSTRUCT.dcm").exists()
            )


if __name__ == "__main__":
    unittest.main()
