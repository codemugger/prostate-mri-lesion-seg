"""DICOM Structured Report export for prostate gland measurements.

The report is a DICOM Comprehensive SR containing a TID 1500 Imaging
Measurement Report. It records measurements derived from the cleaned prostate
mask and references the exact T2 MR series selected by the inference pipeline.

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


def generate_prostate_measurement_sr(
    dicom_series_path: Path,
    output_path: Path,
    measurements: dict,
    cleanup_metrics: dict | None = None,
) -> Path:
    """Create a DICOM SR containing cleaned-prostate measurements.

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
    """
    dicom_series_path = Path(dicom_series_path)
    output_path = Path(output_path)
    evidence_images = _read_reference_images(dicom_series_path)
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
        imaging_measurements=[measurement_group],
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
    document.ContentDescription = "Cleaned prostate dimensions and volume"
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
