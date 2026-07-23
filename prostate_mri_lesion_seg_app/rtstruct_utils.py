'''
Prostate-MRI_Lesion_Detection, v3.0 (Release date: September 17, 2024)
DEFINITIONS: AUTHOR(S) NVIDIA Corp. and National Cancer Institute, NIH

PROVIDER: the National Cancer Institute (NCI), a participating institute of the
National Institutes of Health (NIH), and an agency of the United States Government.

SOFTWARE: the machine readable, binary, object code form,
and the related documentation for the modules of the Prostate-MRI_Lesion_Detection, v2.0
software package, which is a collection of operators which accept (T2, ADC, and High
b-value DICOM images) and produce prostate organ and lesion segmentation files

RECIPIENT: the party that downloads the software.

By downloading or otherwise receiving the SOFTWARE, RECIPIENT may
use and/or redistribute the SOFTWARE, with or without modification,
subject to RECIPIENT's agreement to the following terms:

1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS
OF HUMAN SUBJECTS.  RECIPIENT is responsible for
compliance with all laws and regulations applicable to the use
of the SOFTWARE.

2. THE SOFTWARE is distributed for NON-COMMERCIAL RESEARCH PURPOSES ONLY. RECIPIENT is
responsible for appropriate-use compliance.

3.	RECIPIENT agrees to acknowledge PROVIDER's contribution and
the name of the author of the SOFTWARE in all written publications
containing any data or information regarding or resulting from use
of the SOFTWARE.

4.	THE SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
ARE DISCLAIMED. IN NO EVENT SHALL THE PROVIDER OR THE INDIVIDUAL DEVELOPERS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

5.	RECIPIENT agrees not to use any trademarks, service marks, trade names,
logos or product names of NVIDIA, NCI or NIH to endorse or promote products derived
from the SOFTWARE without specific, prior and written permission.

6.	For sake of clarity, and not by way of limitation, RECIPIENT may add its
own copyright statement to its modifications or derivative works of the SOFTWARE
and may provide additional or different license terms and conditions in its
sublicenses of modifications or derivative works of the SOFTWARE provided that
RECIPIENT's use, reproduction, and distribution of the SOFTWARE otherwise complies
with the conditions stated in this Agreement. Whenever Recipient distributes or
redistributes the SOFTWARE, a copy of this Agreement must be included with
each copy of the SOFTWARE.'''

import os
import copy
import tempfile
import logging
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from scipy import ndimage as ndi
from scipy.ndimage import zoom as scipy_zoom
from typing import Optional

try:
    # External package providing RTStructBuilder
    from rt_utils import RTStructBuilder  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "rt_utils package is required. Install it with: pip install rt-utils"
    ) from exc


def _ensure_study_id(dicom_series_path: str) -> tuple[str, Optional[tempfile.TemporaryDirectory]]:
    """
    Ensure every DICOM file in *dicom_series_path* has the StudyID tag
    (0020,0010).  SGH de-identified data is missing this tag, which causes
    rt_utils / pydicom to crash during RTSTRUCT creation.

    If all files already have StudyID, the original path is returned unchanged.
    Otherwise a temporary copy is made with StudyID populated from
    AccessionNumber (0008,0050) -- matching the expected de-id behaviour.

    Returns (path_to_use, tmpdir_handle).  The caller must keep *tmpdir_handle*
    alive until RTSTRUCT generation is complete; it is ``None`` when no
    patching was required.
    """
    import pydicom

    dcm_files = [
        os.path.join(dicom_series_path, f)
        for f in os.listdir(dicom_series_path)
        if f.lower().endswith(".dcm")
    ]
    if not dcm_files:
        return dicom_series_path, None

    needs_patch = False
    for fp in dcm_files:
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        if (0x0020, 0x0010) not in ds:
            needs_patch = True
            break

    if not needs_patch:
        return dicom_series_path, None

    tmpdir = tempfile.TemporaryDirectory(prefix="rtstruct_patched_")
    patched_count = 0
    for fp in dcm_files:
        ds = pydicom.dcmread(fp, force=True)
        if (0x0020, 0x0010) not in ds:
            accession = getattr(ds, "AccessionNumber", "") or ""
            ds.StudyID = accession[:16]  # VR=SH, max 16 chars
            patched_count += 1
        ds.save_as(os.path.join(tmpdir.name, os.path.basename(fp)))

    logging.info(
        "Patched StudyID (from AccessionNumber) on %d/%d DICOM files in temp dir: %s",
        patched_count, len(dcm_files), tmpdir.name,
    )
    return tmpdir.name, tmpdir


def _prepare_mask_from_array(label_data: np.ndarray) -> np.ndarray:
    """
    Convert a label array into a boolean mask suitable for rt_utils:
    - Promote 2D to 3D if needed
    - Fill holes
    - Transpose (x, y, z) -> (y, x, z)
    """
    mask = label_data.astype(bool)
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    if mask.ndim != 3:
        raise ValueError(f"Expected 2D or 3D mask, got shape {label_data.shape}")

    mask = ndi.binary_fill_holes(mask)

    # Nibabel gives (x, y, z); rt_utils expects (rows, cols, slices) on last axis
    mask = np.transpose(mask, (1, 0, 2))  # (x,y,z) -> (y,x,z)

    return mask  # bool, 3D


def _match_mask_to_dicom_slices(mask: np.ndarray, num_dicom_slices: int) -> np.ndarray:
    """
    Resize the mask's z-dimension (last axis) to match the DICOM series slice
    count using nearest-neighbor interpolation.

    MONAI's DICOMSeriesToVolumeOperator may remove duplicate/problematic DICOM
    slices during volume conversion, so the NIfTI mask can have fewer slices
    than the number of .dcm files on disk.  rt_utils requires an exact match.
    """
    mask_slices = mask.shape[-1]
    if mask_slices == num_dicom_slices:
        return mask

    logging.warning(
        "Mask z-dim (%d) != DICOM slice count (%d). "
        "Resampling mask to match DICOM series.",
        mask_slices, num_dicom_slices,
    )
    zoom_factors = [1.0] * (mask.ndim - 1) + [num_dicom_slices / mask_slices]
    return scipy_zoom(mask.astype(np.float32), zoom_factors, order=0) > 0.5


def _contour_area_mm2(points: np.ndarray) -> float:
    """Return the area of a coplanar 3D polygon using Newell's method."""
    if len(points) < 3:
        return 0.0
    area_vector = 0.5 * np.sum(
        np.cross(points, np.roll(points, -1, axis=0)),
        axis=0,
    )
    return float(np.linalg.norm(area_vector))


def _repair_degenerate_contour(
    contour,
    pixel_axis_0: np.ndarray,
    pixel_axis_1: np.ndarray,
) -> list:
    """Turn a line-like rt-utils contour into valid pixel-footprint polygons.

    OpenCV emits one- or two-point contours for single-pixel and line-like
    components. CLOSEDPLANAR_XOR requires a genuine closed polygon, but these
    tiny components may represent real lesions and must not simply be dropped.
    With contour approximation disabled, the unique points enumerate the thin
    component's pixels. Each is represented by a separate 0.98-pixel square,
    which rasterizes exactly without joining diagonal pixels or adding voxels.
    """
    points = np.asarray(contour.ContourData, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("RTSTRUCT contour contains non-finite coordinates")

    unique_points = np.unique(np.round(points, decimals=8), axis=0)
    if len(unique_points) >= 3 and _contour_area_mm2(points) > 1e-6:
        return [contour]
    if len(unique_points) == 0:
        raise ValueError("RTSTRUCT contour contains no coordinates")

    repaired_contours = []
    for center in unique_points:
        axis_0 = 0.49 * pixel_axis_0
        axis_1 = 0.49 * pixel_axis_1
        repaired = np.asarray(
            [
                center - axis_0 - axis_1,
                center + axis_0 - axis_1,
                center + axis_0 + axis_1,
                center - axis_0 + axis_1,
            ]
        )
        repaired_contour = copy.deepcopy(contour)
        repaired_contour.ContourData = repaired.reshape(-1).tolist()
        repaired_contour.NumberOfContourPoints = len(repaired)
        repaired_contours.append(repaired_contour)
    return repaired_contours


def _finalize_rtstruct_xor_contours(
    rtstruct_path: str | Path,
    series_data,
) -> dict[str, int]:
    """Replace keyholes with standards-based XOR contours and validate them."""
    dataset = pydicom.dcmread(str(rtstruct_path))
    first_slice = series_data[0]
    orientation = np.asarray(
        first_slice.ImageOrientationPatient,
        dtype=np.float64,
    )
    row_spacing, column_spacing = (
        float(value) for value in first_slice.PixelSpacing
    )
    pixel_axis_0 = orientation[:3] * row_spacing
    pixel_axis_1 = orientation[3:] * column_spacing

    repaired_count = 0
    contour_count = 0
    for roi_contour in getattr(dataset, "ROIContourSequence", []):
        finalized_contours = []
        for source_contour in getattr(
            roi_contour,
            "ContourSequence",
            [],
        ):
            repaired_contours = _repair_degenerate_contour(
                source_contour,
                pixel_axis_0,
                pixel_axis_1,
            )
            if (
                len(repaired_contours) != 1
                or repaired_contours[0] is not source_contour
            ):
                repaired_count += 1
            finalized_contours.extend(repaired_contours)

        roi_contour.ContourSequence = finalized_contours
        for contour_number, contour in enumerate(
            finalized_contours,
            start=1,
        ):
            contour.ContourGeometricType = "CLOSEDPLANAR_XOR"
            contour.ContourNumber = contour_number
            points = np.asarray(
                contour.ContourData, dtype=np.float64
            ).reshape(-1, 3)
            if (
                len(points) < 3
                or len(np.unique(np.round(points, decimals=8), axis=0)) < 3
                or _contour_area_mm2(points) <= 1e-6
            ):
                raise ValueError(
                    "Unable to produce a valid closed RTSTRUCT contour"
                )
            contour_count += 1

    dataset.save_as(str(rtstruct_path), enforce_file_format=True)
    logging.info(
        "Finalized %d RTSTRUCT contours as CLOSEDPLANAR_XOR "
        "(%d degenerate contours repaired): %s",
        contour_count,
        repaired_count,
        rtstruct_path,
    )
    return {
        "contour_count": contour_count,
        "repaired_degenerate_contour_count": repaired_count,
    }


def _save_rtstruct_with_xor_contours(rtstruct, output_path: str | Path) -> None:
    """Save an rt-utils object without visible keyhole connector lines."""
    rtstruct.save(str(output_path))
    _finalize_rtstruct_xor_contours(
        output_path,
        rtstruct.series_data,
    )


def load_nifti_mask(nifti_path: str) -> np.ndarray:
    """
    Load a NIfTI file and return a 3D boolean numpy array in (z, y, x) order,
    suitable for rt_utils.

    Methodology matches the notebook:
    1. Load NIfTI and create binary mask (> 0)
    2. Fill holes using binary_fill_holes
    3. Transpose (x,y,z) -> (y,x,z)
    4. Flip rows for RAS (NIfTI) -> LPS (DICOM) alignment (optional)
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata()

    # Binary mask from all non-zero labels
    return _prepare_mask_from_array(data > 0)


def find_dicom_series_by_uid(input_path: Path, series_instance_uid: str) -> Optional[Path]:
    """
    Find the DICOM series folder whose files have the given SeriesInstanceUID.

    Walks *input_path* recursively; for every directory that contains ``.dcm``
    files, reads one file header and compares ``SeriesInstanceUID``.

    Returns the matching directory ``Path``, or ``None``.
    """
    import pydicom

    input_path = Path(input_path)
    if not input_path.exists():
        return None

    for root, _dirs, files in os.walk(input_path):
        dcm_files = [f for f in files if f.lower().endswith(".dcm") and ":" not in f]
        if not dcm_files:
            continue
        sample = os.path.join(root, dcm_files[0])
        try:
            ds = pydicom.dcmread(sample, stop_before_pixels=True, force=True)
            if str(getattr(ds, "SeriesInstanceUID", "")) == series_instance_uid:
                return Path(root)
        except Exception:
            continue

    return None


def find_t2_dicom_series_path(
    input_path: Path,
    series_instance_uid: Optional[str] = None,
) -> Optional[Path]:
    """
    Find the T2 DICOM series folder from the input path.

    If *series_instance_uid* is provided the folder is located by reading one
    DICOM header per subdirectory and matching UIDs exactly.  This is the
    preferred path because it guarantees the RTSTRUCT references the same
    series that the pipeline actually used for inference.

    Falls back to name-based heuristics only when no UID is supplied.
    """
    input_path = Path(input_path)

    if series_instance_uid:
        matched = find_dicom_series_by_uid(input_path, series_instance_uid)
        if matched is not None:
            logging.info("T2 DICOM series matched by UID: %s", matched)
            return matched
        logging.warning(
            "Could not find T2 folder by SeriesInstanceUID %s — "
            "falling back to name heuristics.",
            series_instance_uid,
        )

    if not input_path.exists():
        return None

    t2_patterns = [
        "t2tse",
        "t2_tse",
        "t2tsetra",
        "t2_tse_tra",
        "T2",
        "T2W",
        "T2W_TSE",
        "AX T2",
        "T2 AX",
        "T2 TSE",
    ]

    found_paths: list[tuple[Path, int]] = []

    for root, _dirs, _files in os.walk(input_path):
        root_path = Path(root)
        dir_name = root_path.name.lower()

        for pattern in t2_patterns:
            if pattern.lower() in dir_name:
                dicom_files = list(root_path.glob("*.dcm")) + list(root_path.glob("*.DCM"))
                if len(dicom_files) > 0:
                    if "tra" in dir_name or "t2tsetra" in dir_name:
                        return root_path
                    found_paths.append((root_path, len(dicom_files)))
                    break

    if found_paths:
        found_paths.sort(key=lambda x: x[1], reverse=True)
        return found_paths[0][0]

    for root, _dirs, _files in os.walk(input_path):
        root_path = Path(root)
        dicom_files = list(root_path.glob("*.dcm")) + list(root_path.glob("*.DCM"))
        if len(dicom_files) > 5:
            return root_path

    return None


def create_rtstruct_from_mask(
    dicom_series_path: str,
    nifti_mask_path: str,
    output_rtstruct_path: str,
    roi_name: str = "ROI",
    roi_color: list[int] | None = None,
    use_pin_hole: bool = False,
) -> None:
    """
    Create an RTSTRUCT using separate CLOSEDPLANAR_XOR contours.

    ``use_pin_hole`` is retained for API compatibility but must remain false:
    keyhole encoding introduces visible connector lines through the ROI.
    """
    if roi_color is None:
        roi_color = [255, 0, 0]

    if use_pin_hole:
        raise ValueError(
            "Pinhole RTSTRUCT contours are disabled because they introduce "
            "visible connector lines; use CLOSEDPLANAR_XOR contours instead."
        )
    if not os.path.exists(nifti_mask_path):
        raise FileNotFoundError(f"NIfTI mask file not found: {nifti_mask_path}")

    if not os.path.exists(dicom_series_path):
        raise FileNotFoundError(f"DICOM series path not found: {dicom_series_path}")

    patched_path, tmpdir = _ensure_study_id(dicom_series_path)
    try:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=patched_path)
        num_dicom_slices = len(rtstruct.series_data)

        numpy_segmentation_mask = load_nifti_mask(nifti_mask_path)
        numpy_segmentation_mask = _match_mask_to_dicom_slices(
            numpy_segmentation_mask, num_dicom_slices
        )

        rtstruct.add_roi(
            mask=numpy_segmentation_mask,
            name=roi_name,
            color=roi_color,
            use_pin_hole=False,
            approximate_contours=False,
        )

        _save_rtstruct_with_xor_contours(rtstruct, output_rtstruct_path)
        logging.info("RTSTRUCT saved to: %s", os.path.abspath(output_rtstruct_path))
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def generate_rtstruct_files(
    output_folder: Path,
    input_folder: Path,
    t2_dicom_series_path: Optional[Path] = None,
    t2_series_instance_uid: Optional[str] = None,
) -> dict:
    """
    Generate RTSTRUCT files for organ, lesion, and combined segmentations.

    - Organ (original): organ/organ.nii.gz       -> organ/organ_RTSTRUCT.dcm
    - Organ (cleaned):  organ/cleaned_organ.nii.gz -> organ/cleaned_organ_RTSTRUCT.dcm
    - Lesion:           lesion/lesion_mask.nii.gz -> lesion/lesion_RTSTRUCT.dcm
    - Combined:         cleaned_organ + lesion    -> combined_organ_lesion_RTSTRUCT.dcm

    The combined RTSTRUCT uses the *cleaned* organ mask (islands removed, holes
    filled) so that clinical viewers display only the true prostate contour
    alongside lesion contours.

    *t2_series_instance_uid* (preferred): the DICOM SeriesInstanceUID of the
    T2 series that the pipeline selected.  When provided, the function locates
    the correct DICOM folder by UID, guaranteeing that the RTSTRUCT references
    the same series the viewer will display.
    """
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)

    # Find T2 DICOM series path if not provided
    if t2_dicom_series_path is None:
        t2_dicom_series_path = find_t2_dicom_series_path(
            input_folder, series_instance_uid=t2_series_instance_uid,
        )
        if t2_dicom_series_path is None:
            raise ValueError(
                f"Could not find T2 DICOM series folder in: {input_folder}. "
                "Please provide t2_dicom_series_path explicitly."
            )

    t2_dicom_series_path = Path(t2_dicom_series_path)
    if not t2_dicom_series_path.exists():
        raise FileNotFoundError(f"T2 DICOM series path does not exist: {t2_dicom_series_path}")

    results: dict[str, Optional[str]] = {
        "organ_rtstruct": None,
        "cleaned_organ_rtstruct": None,
        "lesion_rtstruct": None,
        "combined_rtstruct": None,
        "t2_dicom_series": str(t2_dicom_series_path),
    }

    # ---- Organ RTSTRUCT ----
    organ_nifti_path = output_folder / "organ" / "organ.nii.gz"
    if organ_nifti_path.exists():
        organ_rtstruct_path = output_folder / "organ" / "organ_RTSTRUCT.dcm"
        try:
            nii = nib.load(str(organ_nifti_path))
            data = nii.get_fdata()
            max_label = int(data.max())

            patched_path, tmpdir = _ensure_study_id(str(t2_dicom_series_path))
            try:
                rtstruct = RTStructBuilder.create_new(dicom_series_path=patched_path)
                num_dicom_slices = len(rtstruct.series_data)

                if max_label >= 2:
                    # Multi-class organ mask (0=bg, 1=TZ, 2=PZ)
                    tz_mask = _prepare_mask_from_array(data == 1)
                    pz_mask = _prepare_mask_from_array(data == 2)
                    tz_mask = _match_mask_to_dicom_slices(tz_mask, num_dicom_slices)
                    pz_mask = _match_mask_to_dicom_slices(pz_mask, num_dicom_slices)

                    if tz_mask.any():
                        rtstruct.add_roi(
                            mask=tz_mask,
                            name="Prostate_TZ",
                            color=[0, 0, 255],
                            use_pin_hole=False,
                            approximate_contours=False,
                        )

                    if pz_mask.any():
                        rtstruct.add_roi(
                            mask=pz_mask,
                            name="Prostate_PZ",
                            color=[255, 255, 0],
                            use_pin_hole=False,
                            approximate_contours=False,
                        )
                    logging.info("Organ RTSTRUCT created (multi-class): %s", organ_rtstruct_path)
                else:
                    # Binary organ mask (0=bg, 1=prostate)
                    whole_mask = _prepare_mask_from_array(data > 0)
                    whole_mask = _match_mask_to_dicom_slices(whole_mask, num_dicom_slices)
                    if whole_mask.any():
                        rtstruct.add_roi(
                            mask=whole_mask,
                            name="Prostate",
                            color=[0, 255, 0],
                            use_pin_hole=False,
                            approximate_contours=False,
                        )
                    logging.info("Organ RTSTRUCT created (binary): %s", organ_rtstruct_path)

                _save_rtstruct_with_xor_contours(
                    rtstruct,
                    organ_rtstruct_path,
                )
                results["organ_rtstruct"] = str(organ_rtstruct_path)
            finally:
                if tmpdir is not None:
                    tmpdir.cleanup()
        except Exception as exc:  # pragma: no cover - logging path
            logging.error("Failed to create organ RTSTRUCT: %s", exc)
    else:
        logging.warning(
            "Organ NIfTI file not found: %s. Skipping organ RTSTRUCT generation.",
            organ_nifti_path,
        )

    # ---- Cleaned Organ RTSTRUCT ----
    cleaned_organ_nifti_path = output_folder / "organ" / "cleaned_organ.nii.gz"
    if cleaned_organ_nifti_path.exists():
        cleaned_organ_rtstruct_path = output_folder / "organ" / "cleaned_organ_RTSTRUCT.dcm"
        try:
            nii = nib.load(str(cleaned_organ_nifti_path))
            data = nii.get_fdata()
            max_label = int(data.max())

            patched_path, tmpdir = _ensure_study_id(str(t2_dicom_series_path))
            try:
                rtstruct = RTStructBuilder.create_new(dicom_series_path=patched_path)
                num_dicom_slices = len(rtstruct.series_data)

                if max_label >= 2:
                    tz_mask = _prepare_mask_from_array(data == 1)
                    pz_mask = _prepare_mask_from_array(data == 2)
                    tz_mask = _match_mask_to_dicom_slices(tz_mask, num_dicom_slices)
                    pz_mask = _match_mask_to_dicom_slices(pz_mask, num_dicom_slices)

                    if tz_mask.any():
                        rtstruct.add_roi(
                            mask=tz_mask,
                            name="Prostate_TZ",
                            color=[0, 0, 255],
                            use_pin_hole=False,
                            approximate_contours=False,
                        )
                    if pz_mask.any():
                        rtstruct.add_roi(
                            mask=pz_mask,
                            name="Prostate_PZ",
                            color=[255, 255, 0],
                            use_pin_hole=False,
                            approximate_contours=False,
                        )
                    logging.info("Cleaned organ RTSTRUCT created (multi-class): %s", cleaned_organ_rtstruct_path)
                else:
                    whole_mask = _prepare_mask_from_array(data > 0)
                    whole_mask = _match_mask_to_dicom_slices(whole_mask, num_dicom_slices)
                    if whole_mask.any():
                        rtstruct.add_roi(
                            mask=whole_mask,
                            name="Prostate",
                            color=[0, 255, 0],
                            use_pin_hole=False,
                            approximate_contours=False,
                        )
                    logging.info("Cleaned organ RTSTRUCT created (binary): %s", cleaned_organ_rtstruct_path)

                _save_rtstruct_with_xor_contours(
                    rtstruct,
                    cleaned_organ_rtstruct_path,
                )
                results["cleaned_organ_rtstruct"] = str(cleaned_organ_rtstruct_path)
            finally:
                if tmpdir is not None:
                    tmpdir.cleanup()
        except Exception as exc:
            logging.error("Failed to create cleaned organ RTSTRUCT: %s", exc)
    else:
        logging.warning(
            "Cleaned organ NIfTI not found: %s. Skipping cleaned organ RTSTRUCT.",
            cleaned_organ_nifti_path,
        )

    # ---- Lesion RTSTRUCT ----
    lesion_nifti_path = output_folder / "lesion" / "lesion_mask.nii.gz"
    if lesion_nifti_path.exists():
        lesion_rtstruct_path = output_folder / "lesion" / "lesion_RTSTRUCT.dcm"
        try:
            create_rtstruct_from_mask(
                dicom_series_path=str(t2_dicom_series_path),
                nifti_mask_path=str(lesion_nifti_path),
                output_rtstruct_path=str(lesion_rtstruct_path),
                roi_name="Lesion",
                roi_color=[255, 0, 0],  # Red for lesion
                use_pin_hole=False,
            )
            results["lesion_rtstruct"] = str(lesion_rtstruct_path)
            logging.info("Lesion RTSTRUCT created: %s", lesion_rtstruct_path)
        except Exception as exc:  # pragma: no cover - logging path
            logging.error("Failed to create lesion RTSTRUCT: %s", exc)
    else:
        logging.warning(
            "Lesion NIfTI file not found: %s. Skipping lesion RTSTRUCT generation.",
            lesion_nifti_path,
        )

    # ---- Combined Organ + Lesion RTSTRUCT ----
    # Single RTSTRUCT containing all organ and lesion ROIs together, placed in
    # the case root so viewers like CARPL can load one file with every contour.
    # Uses the CLEANED organ mask (islands removed) for clinical accuracy.
    combined_organ_source = output_folder / "organ" / "cleaned_organ.nii.gz"
    if not combined_organ_source.exists():
        # Fallback to original if cleaned doesn't exist
        combined_organ_source = organ_nifti_path
    if combined_organ_source.exists() and lesion_nifti_path.exists():
        combined_rtstruct_path = output_folder / "combined_organ_lesion_RTSTRUCT.dcm"
        try:
            patched_path, tmpdir = _ensure_study_id(str(t2_dicom_series_path))
            try:
                rtstruct = RTStructBuilder.create_new(dicom_series_path=patched_path)
                num_dicom_slices = len(rtstruct.series_data)

                organ_nii = nib.load(str(combined_organ_source))
                organ_data = organ_nii.get_fdata()
                organ_max_label = int(organ_data.max())

                if organ_max_label >= 2:
                    tz_mask = _prepare_mask_from_array(organ_data == 1)
                    pz_mask = _prepare_mask_from_array(organ_data == 2)
                    tz_mask = _match_mask_to_dicom_slices(tz_mask, num_dicom_slices)
                    pz_mask = _match_mask_to_dicom_slices(pz_mask, num_dicom_slices)
                    if tz_mask.any():
                        rtstruct.add_roi(
                            mask=tz_mask, name="Prostate_TZ",
                            color=[0, 0, 255], use_pin_hole=False,
                            approximate_contours=False,
                        )
                    if pz_mask.any():
                        rtstruct.add_roi(
                            mask=pz_mask, name="Prostate_PZ",
                            color=[255, 255, 0], use_pin_hole=False,
                            approximate_contours=False,
                        )
                else:
                    whole_mask = _prepare_mask_from_array(organ_data > 0)
                    whole_mask = _match_mask_to_dicom_slices(whole_mask, num_dicom_slices)
                    if whole_mask.any():
                        rtstruct.add_roi(
                            mask=whole_mask, name="Prostate",
                            color=[0, 255, 0], use_pin_hole=False,
                            approximate_contours=False,
                        )

                lesion_mask = load_nifti_mask(str(lesion_nifti_path))
                lesion_mask = _match_mask_to_dicom_slices(lesion_mask, num_dicom_slices)
                if lesion_mask.any():
                    rtstruct.add_roi(
                        mask=lesion_mask, name="Lesion",
                        color=[255, 0, 0], use_pin_hole=False,
                        approximate_contours=False,
                    )

                _save_rtstruct_with_xor_contours(
                    rtstruct,
                    combined_rtstruct_path,
                )
                results["combined_rtstruct"] = str(combined_rtstruct_path)
                logging.info("Combined organ+lesion RTSTRUCT created: %s", combined_rtstruct_path)
            finally:
                if tmpdir is not None:
                    tmpdir.cleanup()
        except Exception as exc:
            logging.error("Failed to create combined RTSTRUCT: %s", exc)
    else:
        logging.warning(
            "Skipping combined RTSTRUCT: cleaned organ NIfTI exists=%s, lesion NIfTI exists=%s",
            combined_organ_source.exists() if hasattr(combined_organ_source, 'exists') else False,
            lesion_nifti_path.exists(),
        )

    return results


if __name__ == "__main__":  # pragma: no cover - CLI helper
    """
    Standalone usage example:
    python rtstruct_utils.py --output_folder output/ --input_folder test-data/ProstateX-0004/
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate RTSTRUCT files from NIfTI masks")
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to output folder containing organ/ and lesion/ subdirectories",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to input DICOM folder (used to find T2 series)",
    )
    parser.add_argument(
        "--t2_dicom_series",
        type=str,
        default=None,
        help="Optional: Explicit path to T2 DICOM series folder",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    t2_path = Path(args.t2_dicom_series) if args.t2_dicom_series else None
    res = generate_rtstruct_files(
        output_folder=Path(args.output_folder),
        input_folder=Path(args.input_folder),
        t2_dicom_series_path=t2_path,
    )

    print("\nRTSTRUCT Generation Results:")
    print(f"  T2 DICOM Series: {res['t2_dicom_series']}")
    print(f"  Organ RTSTRUCT: {res['organ_rtstruct']}")
    print(f"  Cleaned Organ RTSTRUCT: {res['cleaned_organ_rtstruct']}")
    print(f"  Lesion RTSTRUCT: {res['lesion_rtstruct']}")
    print(f"  Combined RTSTRUCT: {res['combined_rtstruct']}")



