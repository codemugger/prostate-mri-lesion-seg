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

import copy
import logging
import math
from pathlib import Path

# MONAI Deploy SDK imports
from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

# Local imports
from organ_seg_operator import ProstateSegOperator
from custom_lesion_seg_operator import ProstateLesionSegOperator
from custom_lesion_classifier_operator import ProstateLesionClassifierOperator


# ---------------------------------------------------------------------------
# Monkeypatch: fix DICOMSeriesToVolumeOperator.prepare_series
#
# The upstream implementation has two bugs with SGH de-identified DICOM data:
# 1. Slice removal uses enumerate() indices instead of the actual indices,
#    deleting the wrong slices.
# 2. After (broken) removal, surviving instances without `.distance` crash the
#    sort with: AttributeError: 'DICOMSOPInstance' has no attribute 'distance'
# ---------------------------------------------------------------------------
def _patched_prepare_series(self, series):
    if len(series._sop_instances) <= 1:
        series.depth_pixel_spacing = 1.0
        return

    slice_indices_to_be_removed = []
    last_slice_normal = [0.0, 0.0, 0.0]

    for slice_index, slc in enumerate(series._sop_instances):
        distance = 0.0
        point = [0.0, 0.0, 0.0]
        slice_normal = [0.0, 0.0, 0.0]
        slice_position = None
        cosines = None

        try:
            de = slc[0x0020, 0x0037]
            if de is not None:
                cosines = de.value
        except KeyError:
            pass

        try:
            de = slc[0x0020, 0x0032]
            if de is not None:
                slice_position = de.value
        except KeyError:
            pass

        if (cosines is not None) and (slice_position is not None):
            slice_normal[0] = cosines[1] * cosines[5] - cosines[2] * cosines[4]
            slice_normal[1] = cosines[2] * cosines[3] - cosines[0] * cosines[5]
            slice_normal[2] = cosines[0] * cosines[4] - cosines[1] * cosines[3]
            last_slice_normal = copy.deepcopy(slice_normal)

            for i in range(3):
                point[i] = slice_normal[i] * slice_position[i]
            distance = point[0] + point[1] + point[2]

            series._sop_instances[slice_index].distance = distance
            series._sop_instances[slice_index].first_pixel_on_slice_normal = point
        else:
            logging.debug("Removing slice %d: missing orientation/position", slice_index)
            slice_indices_to_be_removed.append(slice_index)

    # Delete in reverse order so indices stay valid
    for idx in reversed(slice_indices_to_be_removed):
        del series._sop_instances[idx]

    if not series._sop_instances:
        raise ValueError("All DICOM slices removed — series has no spatial metadata.")

    # Fallback: if any remaining instance still lacks .distance, sort by InstanceNumber
    if all(hasattr(s, "distance") for s in series._sop_instances):
        series._sop_instances = sorted(series._sop_instances, key=lambda s: s.distance)
    else:
        logging.warning(
            "Some slices missing 'distance'; falling back to InstanceNumber sort."
        )
        series._sop_instances = sorted(
            series._sop_instances,
            key=lambda s: int(getattr(s, "InstanceNumber", 0) or 0),
        )

    series.depth_direction_cosine = copy.deepcopy(last_slice_normal)

    if len(series._sop_instances) > 1:
        p1 = series._sop_instances[0].first_pixel_on_slice_normal
        p2 = series._sop_instances[1].first_pixel_on_slice_normal
        depth_pixel_spacing = math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )
        series.depth_pixel_spacing = depth_pixel_spacing

    s_1 = series._sop_instances[0]
    s_n = series._sop_instances[-1]
    num_slices = len(series._sop_instances)
    self.compute_affine_transform(s_1, s_n, num_slices, series)


DICOMSeriesToVolumeOperator.prepare_series = _patched_prepare_series


# Series selection rules for T2, ADC, and HIGHB.
#
# Derived from Dr. Anh Tuan Doan's manual annotation of ~3,500 SGH cases
# (series_description_sgh_deid_data.tnt_with_anh_input.csv, Apr 2026).
# Validated at 100% precision / 100% recall on 43,470 annotated rows via
# scripts/validate_selection_rules.py.
#
# T2   : must contain "t2" AND (tra|ax|axial); excludes any series with
#        oblique/coronal/sagittal plane, whole-pelvis FOV, TIRM/STIR,
#        SPACE, fat-sat, or any ADC/DWI/diff/TRACEW markers.
# ADC  : contains "adc"; excludes localizer/survey.
# HIGHB: contains any of tracew/dwi/diff/calc/compute(d), OR
#        focus|muse followed by a b-value; excludes any series that also
#        contains "adc" (so ADC+TRACEW series route to ADC, not HIGHB).
#
# ImageType is intentionally NOT required here because SGH data frequently
# lacks this field; the regex alone is strict enough to separate classes.

Rules_T2 = """
{
    "selections": [
        {
            "name": "t2",
            "conditions" :
            {
                "Modality": "MR",
                "SeriesDescription": "^(?!.*(?:obl|oblq|oblique|obq|cor|sag|pelvis|whole[_ ]?pelvis|wholepelvis|tirm|stir|space|localizer|survey|t1|\\\\+c|adc|dwi|diff|tracew?|_fs_|_fs\\\\b|\\\\bfs_|\\\\bfs\\\\b))(?=.*t2)(?=.*(?:tra|\\\\bax\\\\b|axial|^ax[_ ]|_ax[_ ]|AX[_ ])).*"
            }
        }
    ]
}
"""

Rules_ADC = """
{
    "selections": [
        {
            "name": "adc",
            "conditions":
            {
                "Modality": "MR",
                "SeriesDescription": "^(?!.*(?:localizer|survey))(?=.*adc).*"
            }
        }
    ]
}
"""

Rules_HIGHB = """
{
    "selections": [
        {
            "name": "highb",
            "conditions" :
            {
                "Modality": "MR",
                "SeriesDescription": "^(?!.*adc)(?!.*(?:localizer|survey))(?:(?=.*tracew?)|(?=.*dwi)|(?=.*diff)|(?=.*calc)|(?=.*compute[dr]?)|(?=.*(?:focus|muse).*b[0-9])).*"
            }
        }
    ]
}
"""

class AIProstateLesionSegApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""

        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.debug(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.debug(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.debug(f"Begin {self.compose.__name__}")

        # Use command line options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)
        model_path = Path(app_context.model_path)

        self._logger.info(f"App input and output path: {app_input_path}, {app_output_path}")

        # Data pipeline
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="dcm_loader_op"
        )
        series_selector_T2_op = DICOMSeriesSelectorOperator(self, rules=Rules_T2, name="series_selector_T2")
        series_to_vol_T2_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_T2")
        series_selector_ADC_op = DICOMSeriesSelectorOperator(self, rules=Rules_ADC, name="series_selector_ADC")
        series_to_vol_ADC_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_ADC")
        series_selector_HIGHB_op = DICOMSeriesSelectorOperator(self, rules=Rules_HIGHB, name="series_selector_HIGHB")
        series_to_vol_HIGHB_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_HIGHB")

        # AI operators
        # NOTE:
        # Use the MONAI AppContext output_path so that all intermediate NIfTI volumes,
        # segmentations, RTSTRUCTs, and `lesions.txt` are written under the configured
        # output directory (e.g., per‑patient subfolder such as `output/ProstateX-0004/`).
        organ_seg_op = ProstateSegOperator(
            self,
            app_context=app_context,
            model_path=model_path / "organ",
            output_folder=app_output_path,
            name="organ_seg_op",
        )
        lesion_seg_op = ProstateLesionSegOperator(
            self,
            app_context=app_context,
            model_path=model_path,
            output_folder=app_output_path,
            name="lesion_seg_op",
        )
        lesion_classifier_op = ProstateLesionClassifierOperator(
            self,
            app_context=app_context,
            model_path=model_path,
            output_folder=app_output_path,
            name="lesion_classifier_op",
        )

        #################### Pipeline DAG ####################
        # Data ingestion
        self.add_flow(study_loader_op, series_selector_T2_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(study_loader_op, series_selector_ADC_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(study_loader_op, series_selector_HIGHB_op, {("dicom_study_list", "dicom_study_list")})
        self.add_flow(series_selector_T2_op, series_to_vol_T2_op, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(series_selector_ADC_op, series_to_vol_ADC_op, {("study_selected_series_list", "study_selected_series_list")})
        self.add_flow(series_selector_HIGHB_op, series_to_vol_HIGHB_op, {("study_selected_series_list", "study_selected_series_list")})

        # Organ inference
        self.add_flow(series_to_vol_T2_op, organ_seg_op, {("image", "image")})

        # Lesion inference
        self.add_flow(series_to_vol_T2_op, lesion_seg_op, {("image", "image_t2")})
        self.add_flow(series_to_vol_ADC_op, lesion_seg_op, {("image", "image_adc")})
        self.add_flow(series_to_vol_HIGHB_op, lesion_seg_op, {("image", "image_highb")})
        self.add_flow(organ_seg_op, lesion_seg_op, {("seg_image", "image_organ_seg")})

        # Lesion classification
        self.add_flow(series_to_vol_T2_op, lesion_classifier_op, {("image", "image_t2")})
        self.add_flow(series_to_vol_ADC_op, lesion_classifier_op, {("image", "image_adc")})
        self.add_flow(series_to_vol_HIGHB_op, lesion_classifier_op, {("image", "image_highb")})
        self.add_flow(organ_seg_op, lesion_classifier_op, {("seg_image", "image_organ_seg")})
        self.add_flow(lesion_seg_op, lesion_classifier_op, {("seg_image", "image_lesion_seg")})

        self._logger.debug(f"End {self.compose.__name__}")
        #################### Pipeline DAG ####################

if __name__ == "__main__":
    # Creates the app and test it standalone.
    logging.info(f"Begin {__name__}")
    AIProstateLesionSegApp().run()
    logging.info(f"End {__name__}")
