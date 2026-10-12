"""DICOM Structured Report export for prostate measurements and lesion findings.

The report is a DICOM Comprehensive SR containing a TID 1500 Imaging
Measurement Report. It references the exact T2 MR series selected by the
inference pipeline and contains:

* one measurement group for the cleaned prostate (dimensions and volume);
* one group per AI-detected lesion, so the report meets the CARPL DICOM Push
  guide (Method 2, TID 1500): a finding concept (SCT "Lesion" in the
  prostate), a probability, the AI PI-RADS category, and the lesion volume
  and major-axis length from ``lesions.txt``;
* a single "Lesion: Absent" group when a prostate was segmented but no lesion
  was detected.

Lesion numbering reuses the canonical RTSTRUCT lesion mapping
(``urofusion_rtstruct.build_lesion_rois``), so "Lesion N" in this report is
the same lesion as the RTSTRUCT ROI "Lesion N - AI PI-RADS X". The probability
is the peak value of the existing 5-fold ensemble map
(``lesion/merged_lesion_prob.nii.gz``) inside that lesion, as a percentage.
Nothing here changes segmentation, classification or the RTSTRUCT.

The document is intentionally marked COMPLETE but UNVERIFIED and not FINAL:
these are algorithm-generated research results and have not been reviewed or
verified by a physician.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import highdicom as hd
import pydicom
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.uid import (
    EnhancedMRImageStorage,
    LegacyConvertedEnhancedMRImageStorage,
    MRImageStorage,
    generate_uid,
)


LOGGER = logging.getLogger(__name__)

PRIVATE_CODING_SCHEME = "99PMLS"
MR_IMAGE_STORAGE_CLASSES = {
    str(MRImageStorage),
    str(EnhancedMRImageStorage),
    str(LegacyConvertedEnhancedMRImageStorage),
}
DEVICE_UID = "2.25." + str(
    uuid5(NAMESPACE_URL, "prostate-mri-lesion-seg").int
)


def _read_reference_images(dicom_series_path: Path) -> list[Dataset]:
    """Read valid MR image headers from one selected T2 series."""
    candidates = sorted(
        path
        for path in dicom_series_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".dcm"
        and ":" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(
            f"No DICOM files found in selected T2 series: {dicom_series_path}"
        )

    required_uids = (
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
    )
    evidence: list[Dataset] = []
    expected_study_uid = None
    expected_series_uid = None
    for path in candidates:
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "")) != "MR":
            continue
        if not all(getattr(dataset, name, None) for name in required_uids):
            continue
        if str(dataset.SOPClassUID) not in MR_IMAGE_STORAGE_CLASSES:
            continue

        study_uid = str(dataset.StudyInstanceUID)
        series_uid = str(dataset.SeriesInstanceUID)
        if expected_study_uid is None:
            expected_study_uid = study_uid
            expected_series_uid = series_uid
        elif (
            study_uid != expected_study_uid
            or series_uid != expected_series_uid
        ):
            continue

        # highdicom requires these patient/study attributes. SGH de-identified
        # data may omit them or use values longer than the DICOM SH limit.
        accession = str(getattr(dataset, "AccessionNumber", "") or "")
        study_id = str(getattr(dataset, "StudyID", "") or accession)
        dataset.AccessionNumber = accession[:16]
        dataset.StudyID = study_id[:16]
        for attribute in (
            "PatientID",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "StudyDate",
            "StudyTime",
            "ReferringPhysicianName",
        ):
            if not hasattr(dataset, attribute):
                setattr(dataset, attribute, "")
        patient_name = str(dataset.PatientName)
        if patient_name and "^" not in patient_name:
            dataset.PatientName = patient_name + "^"
        evidence.append(dataset)

    if not evidence:
        raise ValueError(
            f"No valid MR evidence object found in: {dicom_series_path}"
        )
    return evidence


def _validated_measurement(measurements: dict, key: str) -> float:
    value = float(measurements[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid prostate measurement {key}={value!r}")
    return value


# Pipeline outputs needed for lesion findings, relative to the case folder.
LESION_INPUTS = (
    "lesions.txt",
    "lesion/lesion_mask.nii.gz",
    "organ/cleaned_organ.nii.gz",
)
LESION_PROBABILITY_MAP = "lesion/merged_lesion_prob.nii.gz"


def _lesion_findings(case_directory: Path) -> dict | None:
    """Collect per-lesion findings from existing pipeline outputs.

    Returns ``None`` when none of the lesion outputs exist (callers that only
    report prostate measurements). Otherwise returns
    ``{"status": "present" | "absent" | "not_assessed", "lesions": [...]}``.
    ``not_assessed`` means no prostate was segmented, so lesion inference was
    skipped and neither presence nor absence is reported.
    """
    present = [rel for rel in LESION_INPUTS if (case_directory / rel).is_file()]
    if not present:
        return None
    if len(present) != len(LESION_INPUTS):
        missing = sorted(set(LESION_INPUTS) - set(present))
        raise FileNotFoundError(f"Incomplete lesion outputs for SR: missing {missing}")

    import nibabel as nib
    import numpy as np
    import yaml
    from urofusion_rtstruct import _binary_mask, _parse_lesion_scores, build_lesion_rois

    organ_image = nib.load(str(case_directory / "organ/cleaned_organ.nii.gz"))
    lesion_image = nib.load(str(case_directory / "lesion/lesion_mask.nii.gz"))
    organ_mask = _binary_mask(organ_image)
    lesion_mask = _binary_mask(lesion_image)
    if not organ_mask.any():
        return {"status": "not_assessed", "lesions": []}
    if not lesion_mask.any():
        return {"status": "absent", "lesions": []}

    probability_path = case_directory / LESION_PROBABILITY_MAP
    if not probability_path.is_file():
        raise FileNotFoundError(f"Lesions present but {LESION_PROBABILITY_MAP} is missing")
    probability = np.asanyarray(nib.load(str(probability_path)).dataobj, dtype=float)
    if probability.ndim == 2:
        probability = probability[:, :, np.newaxis]
    if probability.shape != lesion_mask.shape:
        raise ValueError(
            f"Probability map shape {probability.shape} != lesion mask shape {lesion_mask.shape}"
        )

    lesions_txt = case_directory / "lesions.txt"
    scores = _parse_lesion_scores(lesions_txt)
    rows = yaml.safe_load(lesions_txt.read_text(encoding="utf-8")) or []
    lesion_rows = {
        int(row["Lesion_ID"]): row
        for row in rows
        if isinstance(row, dict) and "Lesion_ID" in row
    }
    # Same mapping and affine as the canonical RTSTRUCT, so lesion numbers agree.
    rois, _mapping = build_lesion_rois(lesion_mask, organ_image.affine, scores)

    lesions = []
    for roi in rois:
        values = probability[roi.mask]
        peak = float(np.max(values)) * 100.0
        if not math.isfinite(peak) or not 0.0 <= peak <= 100.0 + 1e-6:
            raise ValueError(f"Invalid lesion probability {peak!r} for lesion {roi.display_id}")
        row = lesion_rows.get(roi.display_id) if roi.pi_rads is not None else None
        lesions.append(
            {
                "display_id": roi.display_id,
                "pi_rads": roi.pi_rads,
                "peak_probability_pct": round(min(peak, 100.0), 2),
                # Same values as lesions.txt, rounded for display.
                "volume_mm3": round(float(row["Volume"]), 2) if row else None,
                "major_axis_mm": round(float(row["Major_Axis_Length"]), 2) if row else None,
            }
        )
    return {"status": "present", "lesions": lesions}


def _lesion_measurement_groups(
    findings: dict,
    algorithm_id: "hd.sr.AlgorithmIdentification",
    source_images: list,
) -> list:
    """TID 1501 measurement groups for the AI lesion findings."""
    finding_site = hd.sr.FindingSite(codes.SCT.Prostate)
    if findings["status"] == "absent":
        return [
            hd.sr.MeasurementsAndQualitativeEvaluations(
                tracking_identifier=hd.sr.TrackingIdentifier(
                    uid=generate_uid(),
                    identifier="AI lesion detection",
                ),
                finding_type=codes.SCT.Lesion,
                algorithm_id=algorithm_id,
                finding_sites=[finding_site],
                qualitative_evaluations=[
                    hd.sr.QualitativeEvaluation(
                        name=codes.SCT.Lesion,
                        value=codes.SCT.Absent,
                    )
                ],
                source_images=source_images,
            )
        ]

    groups = []
    for lesion in findings["lesions"]:
        measurements = [
            hd.sr.Measurement(
                name=codes.DCM.Probability,
                value=lesion["peak_probability_pct"],
                unit=codes.UCUM.Percent,
                finding_sites=[finding_site],
            )
        ]
        if lesion["volume_mm3"] is not None:
            measurements.append(
                hd.sr.Measurement(
                    name=Code("PMLS005", PRIVATE_CODING_SCHEME, "AI lesion volume"),
                    value=lesion["volume_mm3"],
                    unit=codes.UCUM.CubicMillimeter,
                    finding_sites=[finding_site],
                )
            )
        if lesion["major_axis_mm"] is not None:
            measurements.append(
                hd.sr.Measurement(
                    name=Code("PMLS004", PRIVATE_CODING_SCHEME, "AI lesion major axis length"),
                    value=lesion["major_axis_mm"],
                    unit=codes.UCUM.Millimeter,
                    finding_sites=[finding_site],
                )
            )
        if lesion["pi_rads"] is not None:
            category = Code(
                f"PIRADS{lesion['pi_rads']}",
                PRIVATE_CODING_SCHEME,
                f"PI-RADS {lesion['pi_rads']}",
            )
        else:
            category = Code("PIRADSNA", PRIVATE_CODING_SCHEME, "PI-RADS unavailable")
        groups.append(
            hd.sr.MeasurementsAndQualitativeEvaluations(
                tracking_identifier=hd.sr.TrackingIdentifier(
                    uid=generate_uid(),
                    identifier=f"Lesion {lesion['display_id']}",
                ),
                finding_type=codes.SCT.Lesion,
                algorithm_id=algorithm_id,
                finding_sites=[finding_site],
                measurements=measurements,
                qualitative_evaluations=[
                    hd.sr.QualitativeEvaluation(
                        name=Code("PMLS010", PRIVATE_CODING_SCHEME, "AI PI-RADS assessment category"),
                        value=category,
                    )
                ],
                source_images=source_images,
            )
        )
    return groups


def generate_prostate_measurement_sr(
    dicom_series_path: Path,
    output_path: Path,
    measurements: dict,
    cleanup_metrics: dict | None = None,
    case_directory: Path | None = None,
) -> Path:
    """Create a DICOM SR with prostate measurements and AI lesion findings.

    Parameters
    ----------
    dicom_series_path:
        Folder containing the pipeline-selected T2 DICOM series.
    output_path:
        Destination path, conventionally
        ``<case>/prostate_measurements_SR.dcm``.
    measurements:
        Output from :func:`common.compute_prostate_measurements`.
    cleanup_metrics:
        Optional cleanup audit metrics. Selected values are recorded as
        algorithm parameters in the measurement group.
    case_directory:
        Case output folder holding ``lesions.txt`` and the lesion/organ
        NIfTIs. Defaults to the SR's own folder, which is where the pipeline
        writes it. If none of those outputs exist, only prostate measurements
        are reported.
    """
    dicom_series_path = Path(dicom_series_path)
    output_path = Path(output_path)
    case_directory = Path(case_directory) if case_directory else output_path.parent
    evidence_images = _read_reference_images(dicom_series_path)
    lesion_findings = _lesion_findings(case_directory)
    primary_evidence = evidence_images[0]

    left_right_mm = _validated_measurement(measurements, "left_right_mm")
    anterior_posterior_mm = _validated_measurement(
        measurements,
        "anterior_posterior_mm",
    )
    superior_inferior_mm = _validated_measurement(
        measurements,
        "superior_inferior_mm",
    )
    volume_mm3 = _validated_measurement(measurements, "volume_mm3")
    volume_cc = _validated_measurement(measurements, "volume_cc")

    finding_site = hd.sr.FindingSite(codes.SCT.Prostate)
    method = codes.DCM.ThreeDimensionalMethod
    measurement_items = [
        hd.sr.Measurement(
            name=Code(
                "PMLS001",
                PRIVATE_CODING_SCHEME,
                "Prostate left-right extent",
            ),
            value=left_right_mm,
            unit=codes.UCUM.Millimeter,
            method=method,
            finding_sites=[finding_site],
        ),
        hd.sr.Measurement(
            name=Code(
                "PMLS002",
                PRIVATE_CODING_SCHEME,
                "Prostate anterior-posterior extent",
            ),
            value=anterior_posterior_mm,
            unit=codes.UCUM.Millimeter,
            method=method,
            finding_sites=[finding_site],
        ),
        hd.sr.Measurement(
            name=Code(
                "PMLS003",
                PRIVATE_CODING_SCHEME,
                "Prostate superior-inferior extent",
            ),
            value=superior_inferior_mm,
            unit=codes.UCUM.Millimeter,
            method=method,
            finding_sites=[finding_site],
        ),
        hd.sr.Measurement(
            name=codes.SCT.Volume,
            value=volume_mm3,
            unit=codes.UCUM.CubicMillimeter,
            method=method,
            finding_sites=[finding_site],
        ),
        hd.sr.Measurement(
            name=codes.SCT.Volume,
            value=volume_cc,
            unit=codes.UCUM.CubicCentimeter,
            method=method,
            finding_sites=[finding_site],
        ),
    ]

    algorithm_parameters = [
        "Largest connected component (full 3D connectivity)",
        "3D binary hole filling",
        "Measurements from cleaned NIfTI mask in RAS coordinates",
    ]
    if cleanup_metrics:
        algorithm_parameters.extend(
            [
                "Input components: "
                f"{cleanup_metrics.get('original_component_count', 'unknown')}",
                "Removed islands: "
                f"{cleanup_metrics.get('removed_island_count', 'unknown')}",
            ]
        )

    source_images = [
        hd.sr.SourceImageForMeasurementGroup(
            referenced_sop_class_uid=str(image.SOPClassUID),
            referenced_sop_instance_uid=str(image.SOPInstanceUID),
        )
        for image in evidence_images
    ]
    measurement_group = hd.sr.MeasurementsAndQualitativeEvaluations(
        tracking_identifier=hd.sr.TrackingIdentifier(
            uid=generate_uid(),
            identifier="Cleaned prostate segmentation",
        ),
        finding_type=codes.SCT.Prostate,
        algorithm_id=hd.sr.AlgorithmIdentification(
            name="Prostate MRI Lesion Segmentation",
            version="2026.07",
            parameters=algorithm_parameters,
        ),
        finding_sites=[finding_site],
        measurements=measurement_items,
        source_images=source_images,
    )

    imaging_measurements = [measurement_group]
    if lesion_findings and lesion_findings["status"] in {"present", "absent"}:
        imaging_measurements.extend(
            _lesion_measurement_groups(
                lesion_findings,
                hd.sr.AlgorithmIdentification(
                    name="Prostate MRI Lesion Segmentation",
                    version="2026.07",
                    parameters=[
                        "5-fold 3D RR-UNet ensemble, lesion threshold 0.63",
                        "Probability: peak ensemble voxel probability within the lesion (%)",
                        "PI-RADS: 3D ResNet lesion classifier",
                        "Lesion numbering matches the combined RTSTRUCT ROI names",
                    ],
                ),
                source_images,
            )
        )

    observer_context = hd.sr.ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=(
            hd.sr.DeviceObserverIdentifyingAttributes(
                uid=DEVICE_UID,
                name="Prostate MRI Lesion Segmentation",
                manufacturer_name="NVIDIA and NCI",
                model_name="MONAI Deploy Pipeline",
            )
        ),
    )
    report = hd.sr.MeasurementReport(
        observation_context=hd.sr.ObservationContext(
            observer_device_context=observer_context,
        ),
        procedure_reported=Code(
            "433455006",
            "SCT",
            "Multiparametric MRI of prostate",
        ),
        imaging_measurements=imaging_measurements,
        title=codes.DCM.ImagingMeasurementReport,
        referenced_images=evidence_images,
    )

    document = hd.sr.ComprehensiveSR(
        evidence=evidence_images,
        content=report,
        series_instance_uid=generate_uid(),
        series_number=900,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="NVIDIA and NCI",
        institution_name=(
            str(primary_evidence.InstitutionName)
            if getattr(primary_evidence, "InstitutionName", None)
            else None
        ),
        is_complete=True,
        is_final=False,
        is_verified=False,
    )
    document.SeriesDescription = "AI Prostate Measurements"
    document.ContentDescription = (
        "Prostate measurements and AI lesion findings"
        if lesion_findings and lesion_findings["status"] in {"present", "absent"}
        else "Cleaned prostate dimensions and volume"
    )
    document.ContentCreatorName = "PROSTATE^MRI^AI"

    private_scheme = Dataset()
    private_scheme.CodingSchemeDesignator = PRIVATE_CODING_SCHEME
    private_scheme.CodingSchemeName = "Prostate MRI Lesion Segmentation"
    private_scheme.CodingSchemeResponsibleOrganization = "NVIDIA and NCI"
    document.CodingSchemeIdentificationSequence = [private_scheme]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save_as(str(output_path), enforce_file_format=True)

    # Round-trip validation catches malformed files before they reach the
    # output directory or a clinical viewer.
    saved = pydicom.dcmread(str(output_path), stop_before_pixels=True)
    if str(saved.Modality) != "SR":
        raise ValueError(f"Generated object is not DICOM SR: {output_path}")
    if str(saved.StudyInstanceUID) != str(primary_evidence.StudyInstanceUID):
        raise ValueError("DICOM SR StudyInstanceUID does not match T2 evidence")
    hd.sr.ComprehensiveSR.from_dataset(saved)

    LOGGER.info("DICOM prostate measurement SR saved: %s", output_path)
    return output_path
