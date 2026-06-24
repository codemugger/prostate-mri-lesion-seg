"""Filter a HIGHB DICOMSeries to retain only slices at the highest b-value.

Background
----------
SGH multi-b DWI series (e.g., ``ep2d_diff_b50_500_1000_1800 prostate_TRACEW``)
pack all b-value volumes into a single DICOM series.  A 24-slice acquisition
at 4 b-values produces 96 DICOM instances.  MONAI Deploy's
``DICOMSeriesToVolumeOperator`` blindly stacks every instance into one 3-D
volume, producing a (96, H, W) array whose slices alternate between
b50 / b500 / b1000 / b1800.  The downstream ``SegmentationDataset`` then
resamples this to match T2 dimensions, spatially averaging across b-values —
producing a physically meaningless input for the lesion model.

This operator is inserted **between** ``DICOMSeriesSelectorOperator`` and
``DICOMSeriesToVolumeOperator`` in the HIGHB branch **only**.  It:

1. Reads each SOP instance's b-value (standard tag ``(0018,9087)`` first,
   Siemens private tag ``(0019,100C)`` as fallback).
2. Identifies the maximum b-value.
3. Removes all SOP instances that do not carry the maximum b-value.
4. Passes the pruned series forward so the volume converter stacks only
   the highest-b slices.

If no b-value tags are found (e.g., some GE sequences), the series passes
through unmodified — safe default.

B-values are rounded to the nearest integer before comparison to handle
floating-point storage (e.g., 1799.99 → 1800).
"""

import logging
from typing import List, Optional

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries

_STANDARD_B_VALUE_TAG = (0x0018, 0x9087)
_SIEMENS_B_VALUE_TAG = (0x0019, 0x100C)


def get_dicom_bvalue(sop_or_dataset) -> Optional[float]:
    """Extract b-value from a DICOMSOPInstance or pydicom Dataset.

    Checks the standard DICOM ``DiffusionBValue`` tag ``(0018,9087)`` first,
    then falls back to the Siemens private tag ``(0019,100C)``.

    Returns the b-value as a float rounded to the nearest integer,
    or ``None`` if neither tag is present.
    """
    native = (
        sop_or_dataset.get_native_sop_instance()
        if hasattr(sop_or_dataset, "get_native_sop_instance")
        else sop_or_dataset
    )

    for tag in (_STANDARD_B_VALUE_TAG, _SIEMENS_B_VALUE_TAG):
        try:
            de = native[tag]
            if de is not None and de.value is not None:
                val = de.value
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    val = list(val)[0]
                return round(float(val))
        except (KeyError, ValueError, TypeError, IndexError):
            continue

    return None


class HighBValueFilterOperator(Operator):
    """Filters a HIGHB series in-place to retain only the highest b-value slices.

    Named input:
        study_selected_series_list: List[StudySelectedSeries] (from selector).
    Named output:
        study_selected_series_list: List[StudySelectedSeries] (filtered).
    """

    def __init__(self, fragment: Fragment, *args, **kwargs) -> None:
        self.input_name = "study_selected_series_list"
        self.output_name = "study_selected_series_list"
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name)
        spec.output(self.output_name).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        study_selected_series_list = op_input.receive(self.input_name)

        if study_selected_series_list:
            for study_ss in study_selected_series_list:
                for selected_series in study_ss.selected_series:
                    self._filter_to_highest_bvalue(selected_series.series)

        op_output.emit(study_selected_series_list, self.output_name)

    def _filter_to_highest_bvalue(self, series: DICOMSeries) -> None:
        """Remove SOP instances whose b-value is not the series maximum."""
        sop_instances = series._sop_instances
        if not sop_instances:
            return

        instance_bvalues = [get_dicom_bvalue(sop) for sop in sop_instances]

        known = {b for b in instance_bvalues if b is not None}

        if not known:
            self._logger.info(
                "HIGHB series (%s): no b-value tags found on %d instances — "
                "passing through unfiltered.",
                series.SeriesInstanceUID,
                len(sop_instances),
            )
            return

        if len(known) <= 1:
            self._logger.info(
                "HIGHB series (%s): single b-value b=%.0f across %d instances — "
                "no filtering needed.",
                series.SeriesInstanceUID,
                next(iter(known)),
                len(sop_instances),
            )
            return

        max_bval = max(known)
        discarded = sorted(known - {max_bval})

        filtered = [
            sop
            for sop, bval in zip(sop_instances, instance_bvalues)
            if bval == max_bval
        ]

        series._sop_instances = filtered

        self._logger.info(
            "HIGHB series (%s): kept %d/%d slices at b=%.0f, "
            "discarded b-values: %s",
            series.SeriesInstanceUID,
            len(filtered),
            len(sop_instances),
            max_bval,
            discarded,
        )
        print(
            f"[HighBValueFilter] Kept {len(filtered)}/{len(sop_instances)} "
            f"slices at b={max_bval:.0f}, discarded b-values: {discarded}"
        )
