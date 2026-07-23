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
subject to RECIPIENT’s agreement to the following terms:

1. THE SOFTWARE SHALL NOT BE USED IN THE TREATMENT OR DIAGNOSIS
OF HUMAN SUBJECTS.  RECIPIENT is responsible for
compliance with all laws and regulations applicable to the use
of the SOFTWARE.

2. THE SOFTWARE is distributed for NON-COMMERCIAL RESEARCH PURPOSES ONLY. RECIPIENT is
responsible for appropriate-use compliance.

3.	RECIPIENT agrees to acknowledge PROVIDER’s contribution and
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
RECIPIENT’s use, reproduction, and distribution of the SOFTWARE otherwise complies
with the conditions stated in this Agreement. Whenever Recipient distributes or
redistributes the SOFTWARE, a copy of this Agreement must be included with
each copy of the SOFTWARE.'''

import logging
import numpy as np
import copy
from skimage.measure import label
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    label as ndimage_label,
)

###############################################################################
def bounding_box_3d(nda):

    r = np.any(nda, axis=(1, 2))
    c = np.any(nda, axis=(0, 2))
    z = np.any(nda, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    # values are the indices with non-zero entries
    bbox = np.zeros(shape=(6,), dtype=np.int16)
    bbox[0] = rmin
    bbox[1] = rmax
    bbox[2] = cmin
    bbox[3] = cmax
    bbox[4] = zmin
    bbox[5] = zmax

    return bbox

def crop_pos_classification_multi_channel_3d(nda, nda_gt, crop_size):
    if nda_gt.ndim == 4:
        nda_ref = np.amax(nda_gt, axis=0)
    elif nda_gt.ndim == 3:
        nda_ref = copy.deepcopy(nda_gt)

    if np.amax(nda_gt) < 1:
        return nda, []

    label_image = label(nda_gt)
    num_crop = len(np.unique(label_image)) - 1

    if num_crop == 1:
        bbox = bounding_box_3d(label_image == 1)

        indices = np.zeros(shape=(3,), dtype=np.int16)
        for j in range(3):
            indices[j] = 0.5 * (bbox[2 * j] + bbox[2 * j + 1])
            indices[j] = indices[j] - int(float(crop_size[j]) / 2.0)
            indices[j] = np.maximum(indices[j], 0)
            indices[j] = np.minimum(indices[j], nda_ref.shape[j] - crop_size[j])

        nda = nda[
            ...,
            indices[0]:indices[0] + crop_size[0],
            indices[1]:indices[1] + crop_size[1],
            indices[2]:indices[2] + crop_size[2]
            ]

        gt = np.unique(nda_gt[label_image==1])[0]
    elif num_crop > 1:
        images = []
        labels = []
        for k in range(num_crop):
            bbox = bounding_box_3d(label_image == k + 1)

            indices = np.zeros(shape=(3,), dtype=np.int16)
            for j in range(3):
                indices[j] = 0.5 * (bbox[2 * j] + bbox[2 * j + 1])
                indices[j] = indices[j] - int(float(crop_size[j]) / 2.0)
                indices[j] = np.maximum(indices[j], 0)
                indices[j] = np.minimum(indices[j], nda_ref.shape[j] - crop_size[j])

            nda_crop = nda[
                ...,
                indices[0]:indices[0] + crop_size[0],
                indices[1]:indices[1] + crop_size[1],
                indices[2]:indices[2] + crop_size[2]
                ]

            gt = np.unique(nda_gt[label_image == k + 1])[0]

            images.append(nda_crop)
            labels.append(gt)
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        return images, labels

    return nda, gt

def standard_normalization_multi_channel(nda):
    for _i in range(nda.shape[0]):
        if np.amax(np.abs(nda[_i, ...])) < 1e-7:
            continue
        nda[_i, ...] = (nda[_i, ...] - np.mean(nda[_i, ...])) / np.std(nda[_i, ...])

    return nda


###############################################################################
# Organ mask cleanup: remove erroneous islands, fill holes
###############################################################################

def clean_organ_mask(
    organ_data: np.ndarray,
    fill_holes: bool = True,
    return_metrics: bool = False,
):
    """Clean a multi-class organ segmentation mask by removing erroneous islands.

    Strategy:
      1. Binarize the organ mask (any non-zero label → 1).
      2. Keep only the largest connected component (the prostate gland).
      3. Optionally fill holes in the largest component for smoother contours.
      4. Restore the original class labels (0=background, 1=TZ, 2=PZ).
         Newly filled voxels receive the nearest retained non-zero label.

    This handles the multi-class case correctly: a TZ island disconnected from
    the main prostate mass will be removed even though its label differs from
    PZ voxels. The spatial connectivity is what matters, not the label value.

    Parameters
    ----------
    organ_data : np.ndarray
        3D array with class labels (0=background, 1=TZ, 2=PZ or 0=bg, 1=prostate).
    fill_holes : bool
        If True, apply binary_fill_holes to the cleaned binary mask before
        restoring class labels. This fills internal cavities.
    return_metrics : bool
        If True, return ``(cleaned_mask, metrics)``. The metrics are intended
        for per-case audit logging.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, dict]
        Cleaned organ mask with the same dtype and shape as input, optionally
        accompanied by cleanup metrics.
    """
    if organ_data is None:
        raise ValueError("organ_data must not be None")
    if organ_data.ndim < 2:
        raise ValueError(
            f"Expected a 2D or 3D organ mask, got shape {organ_data.shape}"
        )

    original_dtype = organ_data.dtype
    binary_mask = (organ_data > 0).astype(np.uint8)
    original_voxels = int(np.sum(binary_mask))

    metrics = {
        "cleanup_applied": True,
        "connectivity": "full",
        "original_component_count": 0,
        "original_foreground_voxels": original_voxels,
        "largest_component_voxels": 0,
        "largest_component_fraction": 0.0,
        "removed_island_count": 0,
        "removed_voxels": 0,
        "filled_hole_voxels": 0,
        "cleaned_foreground_voxels": 0,
    }

    if original_voxels == 0:
        cleaned_empty = organ_data.copy()
        return (
            (cleaned_empty, metrics)
            if return_metrics
            else cleaned_empty
        )

    # Full connectivity (8-connected in 2D, 26-connected in 3D) avoids
    # incorrectly splitting voxels that meet diagonally.
    structure = np.ones((3,) * binary_mask.ndim, dtype=np.uint8)
    labeled_array, num_features = ndimage_label(
        binary_mask,
        structure=structure,
    )
    component_sizes = np.bincount(labeled_array.ravel())
    largest_label = int(np.argmax(component_sizes[1:]) + 1)
    largest_voxels = int(component_sizes[largest_label])
    largest_binary = labeled_array == largest_label

    metrics["original_component_count"] = int(num_features)
    metrics["largest_component_voxels"] = largest_voxels
    metrics["largest_component_fraction"] = round(
        largest_voxels / original_voxels,
        6,
    )
    metrics["removed_island_count"] = max(int(num_features) - 1, 0)
    metrics["removed_voxels"] = original_voxels - largest_voxels

    cleaned_binary = largest_binary
    if fill_holes:
        cleaned_binary = binary_fill_holes(largest_binary)
    newly_filled = cleaned_binary & ~largest_binary
    metrics["filled_hole_voxels"] = int(np.sum(newly_filled))
    metrics["cleaned_foreground_voxels"] = int(np.sum(cleaned_binary))

    # Preserve existing TZ/PZ labels on the retained component.
    cleaned_organ = np.zeros_like(organ_data)
    cleaned_organ[largest_binary] = organ_data[largest_binary]

    # Fill newly created voxels with the nearest retained class label. This is
    # necessary for multi-class masks: merely filling the binary mask would
    # otherwise leave the original zero-valued holes unchanged.
    if np.any(newly_filled):
        _, nearest_indices = distance_transform_edt(
            cleaned_organ == 0,
            return_indices=True,
        )
        nearest_labels = cleaned_organ[tuple(nearest_indices)]
        cleaned_organ[newly_filled] = nearest_labels[newly_filled]

    removed_voxels = metrics["removed_voxels"]
    if removed_voxels or metrics["filled_hole_voxels"]:
        logging.info(
            "Organ cleanup: components=%d, removed_islands=%d, "
            "removed_voxels=%d, filled_hole_voxels=%d, "
            "largest_fraction=%.4f.",
            metrics["original_component_count"],
            metrics["removed_island_count"],
            removed_voxels,
            metrics["filled_hole_voxels"],
            metrics["largest_component_fraction"],
        )
    else:
        logging.info(
            "Organ cleanup: mask already clean (one component, no holes)."
        )

    cleaned_organ = cleaned_organ.astype(original_dtype)
    return (
        (cleaned_organ, metrics)
        if return_metrics
        else cleaned_organ
    )


def clean_lesion_mask(
    lesion_data: np.ndarray,
    min_voxels: int = 10,
    voxel_volume_mm3: float = 0.125,
) -> np.ndarray:
    """Remove tiny lesion islands below a minimum voxel count threshold.

    Unlike organ cleanup, we do NOT keep only the largest component —
    patients can have multiple legitimate lesion foci. Instead, we remove
    only islands smaller than *min_voxels* which are almost certainly noise
    or false-positive fragmentary detections.

    Parameters
    ----------
    lesion_data : np.ndarray
        3D binary lesion mask (0=background, 1=lesion).
    min_voxels : int
        Minimum number of voxels for a lesion island to be retained.
        Default 10 voxels at 0.5mm isotropic = 1.25 mm³ minimum volume.
    voxel_volume_mm3 : float
        Volume of a single voxel in mm³ (for logging only). Default 0.125
        corresponds to 0.5×0.5×0.5 mm isotropic.

    Returns
    -------
    np.ndarray
        Cleaned lesion mask with same dtype and shape as input.
    """
    if lesion_data is None or not np.any(lesion_data):
        return lesion_data

    original_dtype = lesion_data.dtype
    binary_mask = (lesion_data > 0).astype(np.uint8)

    structure = np.ones((3,) * binary_mask.ndim, dtype=np.uint8)
    labeled_array, num_features = ndimage_label(
        binary_mask,
        structure=structure,
    )
    if num_features <= 1:
        # Single island or empty — check size
        if num_features == 1:
            size = int(np.sum(binary_mask))
            if size < min_voxels:
                logging.info(
                    "Lesion cleanup: removed single island of %d voxels (< %d threshold).",
                    size, min_voxels,
                )
                return np.zeros_like(lesion_data)
        return lesion_data

    # Compute sizes of each component
    component_sizes = np.bincount(labeled_array.ravel())
    # component_sizes[0] is background

    removed_count = 0
    removed_voxels = 0
    cleaned_mask = binary_mask.copy()

    for label_id in range(1, num_features + 1):
        size = component_sizes[label_id]
        if size < min_voxels:
            cleaned_mask[labeled_array == label_id] = 0
            removed_count += 1
            removed_voxels += size

    if removed_count > 0:
        min_vol_mm3 = min_voxels * voxel_volume_mm3
        logging.info(
            "Lesion cleanup: removed %d island(s) totalling %d voxels "
            "(each < %d voxels / %.2f mm³). Remaining islands: %d.",
            removed_count, removed_voxels, min_voxels, min_vol_mm3,
            num_features - removed_count,
        )
    else:
        logging.info(
            "Lesion cleanup: all %d island(s) above threshold (%d voxels).",
            num_features, min_voxels,
        )

    return cleaned_mask.astype(original_dtype)


###############################################################################
# Prostate volume and orthogonal dimensions calculation
###############################################################################

def compute_prostate_measurements(
    organ_data: np.ndarray,
    voxel_spacing: tuple | None = None,
    affine: np.ndarray | None = None,
) -> dict:
    """Compute prostate gland physical measurements from a cleaned organ mask.

    Parameters
    ----------
    organ_data : np.ndarray
        3D organ mask (cleaned). Any non-zero voxel is considered prostate.
    voxel_spacing : tuple of float, optional
        Fallback ``(spacing_x, spacing_y, spacing_z)`` in mm. Used only when
        *affine* is not provided.
    affine : np.ndarray, optional
        NIfTI voxel-to-RAS affine. When provided, volume is calculated from
        its determinant and the three dimensions are reported along patient
        left-right (LR), anterior-posterior (AP), and superior-inferior (SI)
        axes, including the full physical voxel footprint.

    Returns
    -------
    dict with keys:
        - volume_mm3: float — total prostate volume in cubic millimeters
        - volume_cc: float — total prostate volume in cubic centimeters (mL)
        - left_right_mm: float — physical LR extent
        - anterior_posterior_mm: float — physical AP extent
        - superior_inferior_mm: float — physical SI extent
        - dim_x_mm/dim_y_mm/dim_z_mm: aliases for LR/AP/SI
        - dimensions_mm: list of 3 floats sorted descending [largest, middle, smallest]
    """
    result = {
        "volume_mm3": 0.0,
        "volume_cc": 0.0,
        "left_right_mm": 0.0,
        "anterior_posterior_mm": 0.0,
        "superior_inferior_mm": 0.0,
        "dim_x_mm": 0.0,
        "dim_y_mm": 0.0,
        "dim_z_mm": 0.0,
        "dimensions_mm": [0.0, 0.0, 0.0],
    }

    if organ_data is None or not np.any(organ_data):
        return result

    binary_mask = organ_data > 0
    if binary_mask.ndim == 2:
        binary_mask = binary_mask[..., np.newaxis]
    if binary_mask.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D organ mask, got shape {organ_data.shape}"
        )

    prostate_voxels = int(np.sum(binary_mask))
    coords = np.argwhere(binary_mask > 0)
    if len(coords) == 0:
        return result

    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValueError(
                f"Expected a 4x4 affine matrix, got shape {affine.shape}"
            )
        linear = affine[:3, :3]
        voxel_volume_mm3 = abs(float(np.linalg.det(linear)))

        # NIfTI world coordinates are RAS: X=LR, Y=AP, Z=SI. Transform
        # foreground voxel centers and then add the projected half-voxel
        # footprint on each world axis. This gives the physical bounding-box
        # extent without assuming voxel axes are already canonical.
        world_centers = coords @ linear.T + affine[:3, 3]
        center_extent = np.ptp(world_centers, axis=0)
        half_voxel_extent = 0.5 * np.sum(np.abs(linear), axis=1)
        physical_extents = center_extent + 2.0 * half_voxel_extent
    else:
        if voxel_spacing is None:
            raise ValueError("Either affine or voxel_spacing must be provided")
        spacing = np.asarray(voxel_spacing[:3], dtype=np.float64)
        if spacing.shape != (3,) or np.any(spacing <= 0):
            raise ValueError(
                f"Invalid voxel spacing: {voxel_spacing}"
            )
        voxel_volume_mm3 = float(np.prod(spacing))
        extent_voxels = coords.max(axis=0) - coords.min(axis=0) + 1
        physical_extents = extent_voxels * spacing

    volume_mm3 = prostate_voxels * voxel_volume_mm3
    volume_cc = volume_mm3 / 1000.0
    left_right_mm, anterior_posterior_mm, superior_inferior_mm = [
        float(value) for value in physical_extents
    ]
    dimensions = sorted(
        [
            left_right_mm,
            anterior_posterior_mm,
            superior_inferior_mm,
        ],
        reverse=True,
    )

    result["volume_mm3"] = round(volume_mm3, 2)
    result["volume_cc"] = round(volume_cc, 2)
    result["left_right_mm"] = round(left_right_mm, 2)
    result["anterior_posterior_mm"] = round(anterior_posterior_mm, 2)
    result["superior_inferior_mm"] = round(superior_inferior_mm, 2)
    result["dim_x_mm"] = result["left_right_mm"]
    result["dim_y_mm"] = result["anterior_posterior_mm"]
    result["dim_z_mm"] = result["superior_inferior_mm"]
    result["dimensions_mm"] = [round(d, 2) for d in dimensions]

    return result
