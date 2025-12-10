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
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage as ndi
from typing import Optional

try:
    # External package providing RTStructBuilder
    from rt_utils import RTStructBuilder  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "rt_utils package is required. Install it with: pip install rt-utils"
    ) from exc


def _prepare_mask_from_array(label_data: np.ndarray) -> np.ndarray:
    """
    Convert a label array into a boolean mask suitable for rt_utils:
    - Fill holes
    - Transpose (x, y, z) -> (y, x, z)
    """
    mask = label_data.astype(bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")

    # Fill holes slice-wise in 3D
    mask = ndi.binary_fill_holes(mask)

    # Nibabel gives (x, y, z); rt_utils expects (rows, cols, slices) on last axis
    mask = np.transpose(mask, (1, 0, 2))  # (x,y,z) -> (y,x,z)

    # RAS (NIfTI) -> LPS (DICOM): flip row axis if needed.
    # For now no need to flip, because the DICOM series is already in LPS orientation.
    # mask = mask[::-1, :, :]

    return mask  # bool, 3D


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


def find_t2_dicom_series_path(input_path: Path) -> Optional[Path]:
    """
    Find the T2 DICOM series folder from the input path.

    Uses name heuristics consistent with the ProstateX data and Rules_T2.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        return None

    # Common T2 series folder name patterns (from app.py Rules_T2 and notebook example)
    # The notebook uses: "5.000000-t2tsetra-75680"
    t2_patterns = [
        "t2tse",
        "t2_tse",
        "t2tsetra",
        "t2_tse_tra",
        "t2tsecor",
        "t2tsesag",
        "T2",
        "T2W",
        "T2W_TSE",
        "AX T2",
        "T2 AX",
        "T2 TSE",
        "t2_tse",
    ]

    # Search recursively for T2 series folders
    found_paths: list[tuple[Path, int]] = []

    for root, _dirs, _files in os.walk(input_path):
        root_path = Path(root)
        dir_name = root_path.name.lower()

        # Check if directory name contains T2 pattern
        for pattern in t2_patterns:
            if pattern.lower() in dir_name:
                dicom_files = list(root_path.glob("*.dcm")) + list(root_path.glob("*.DCM"))
                if len(dicom_files) > 0:
                    # Prefer "tra" (transverse) T2 series as it's most common for prostate
                    if "tra" in dir_name or "t2tsetra" in dir_name:
                        return root_path
                    found_paths.append((root_path, len(dicom_files)))
                    break

    # If we found T2 folders, return the one with most DICOM files
    if found_paths:
        found_paths.sort(key=lambda x: x[1], reverse=True)
        return found_paths[0][0]

    # Fallback: any directory with several DICOM slices
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
    use_pin_hole: bool = True,
) -> None:
    """
    Create an RTSTRUCT file from a NIfTI mask, mirroring the notebook pipeline.
    """
    if roi_color is None:
        roi_color = [255, 0, 0]

    if not os.path.exists(nifti_mask_path):
        raise FileNotFoundError(f"NIfTI mask file not found: {nifti_mask_path}")

    if not os.path.exists(dicom_series_path):
        raise FileNotFoundError(f"DICOM series path not found: {dicom_series_path}")

    # Reference DICOM T2 series
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)

    # Load segmentation (boolean, (z,y,x))
    numpy_segmentation_mask = load_nifti_mask(nifti_mask_path)

    # Add as ROI
    rtstruct.add_roi(
        mask=numpy_segmentation_mask,
        name=roi_name,
        color=roi_color,
        use_pin_hole=use_pin_hole,
    )

    rtstruct.save(output_rtstruct_path)
    logging.info("RTSTRUCT saved to: %s", os.path.abspath(output_rtstruct_path))


def generate_rtstruct_files(
    output_folder: Path,
    input_folder: Path,
    t2_dicom_series_path: Optional[Path] = None,
) -> dict:
    """
    Generate RTSTRUCT files for both organ and lesion segmentations.

    - Organ:  organ/organ.nii.gz   -> organ_RTSTRUCT.dcm
    - Lesion: lesion/lesion_mask.nii.gz -> lesion_RTSTRUCT.dcm
    """
    output_folder = Path(output_folder)
    input_folder = Path(input_folder)

    # Find T2 DICOM series path if not provided
    if t2_dicom_series_path is None:
        t2_dicom_series_path = find_t2_dicom_series_path(input_folder)
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
        "lesion_rtstruct": None,
        "t2_dicom_series": str(t2_dicom_series_path),
    }

    # ---- Organ RTSTRUCT ----
    organ_nifti_path = output_folder / "organ" / "organ.nii.gz"
    if organ_nifti_path.exists():
        organ_rtstruct_path = output_folder / "organ" / "organ_RTSTRUCT.dcm"
        try:
            # Load multi-class organ mask (0=bg, 1=TZ, 2=PZ) and create:
            # - Whole prostate ROI (TZ+PZ)
            # - TZ ROI (label==1)
            # - PZ ROI (label==2)
            nii = nib.load(str(organ_nifti_path))
            data = nii.get_fdata()

            whole_mask = _prepare_mask_from_array(data > 0)
            tz_mask = _prepare_mask_from_array(data == 1)
            pz_mask = _prepare_mask_from_array(data == 2)

            # Create a single RTSTRUCT with multiple ROIs for organ zones.
            rtstruct = RTStructBuilder.create_new(dicom_series_path=str(t2_dicom_series_path))

            # Whole prostate (union of TZ+PZ) – keeps backward-compatible organ concept.
            if whole_mask.any():
                rtstruct.add_roi(
                    mask=whole_mask,
                    name="Prostate_Organ",
                    color=[0, 255, 0],  # Green
                    use_pin_hole=True,
                )

            # Transition Zone (TZ)
            if tz_mask.any():
                rtstruct.add_roi(
                    mask=tz_mask,
                    name="Prostate_TZ",
                    color=[0, 0, 255],  # Blue
                    use_pin_hole=True,
                )

            # Peripheral Zone (PZ)
            if pz_mask.any():
                rtstruct.add_roi(
                    mask=pz_mask,
                    name="Prostate_PZ",
                    color=[255, 255, 0],  # Yellow
                    use_pin_hole=True,
                )

            rtstruct.save(str(organ_rtstruct_path))
            results["organ_rtstruct"] = str(organ_rtstruct_path)
            logging.info("Organ RTSTRUCT created (multi-class): %s", organ_rtstruct_path)
        except Exception as exc:  # pragma: no cover - logging path
            logging.error("Failed to create organ RTSTRUCT: %s", exc)
    else:
        logging.warning(
            "Organ NIfTI file not found: %s. Skipping organ RTSTRUCT generation.",
            organ_nifti_path,
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
                use_pin_hole=True,
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
    print(f"  Lesion RTSTRUCT: {res['lesion_rtstruct']}")



