# Draft update for Dr Anh — UroFusion RTSTRUCT and CARPL series filter

This is a draft for human review before sending. No message has been sent to
Dr Anh or CARPL.

## Suggested reply to Dr Anh

Hi Anh, thanks for the reminder. I have completed another guarded UroFusion
review update and prepared the CARPL series-filter requirements below.

The lesions are now separate RTSTRUCT ROIs and are labelled independently with
their AI PI-RADS results. I also traced the multiple green prostate outlines in
the fragmented case: they were not duplicate Prostate ROIs; they were nested
contours generated from enclosed holes and small disconnected regions in the
source gland mask.

For the new review RTSTRUCTs, the prostate is exported as exactly one filled
outer contour on each occupied T2 slice. The exporter now rejects a prostate
RTSTRUCT if more than one prostate contour remains on any slice. This is an
isolated display-compatibility profile only—the original cleaned gland NIfTI,
main RTSTRUCTs, SR, `lesions.txt`, and inference completion status are unchanged.

The prostate smoothing remains deterministic 3D physical-space smoothing at
1.0 mm. I increased the permitted absolute volume change from 2% to 8% as
discussed, but retained the other safety gates: same apex/base slices, no new
holes, one 3D component, maximum surface displacement no more than 2 mm, and
100% containment of the unchanged lesion mask.

I have prepared nine review cases in `28_july_outputs_follow_up_for_anh`, with
control and smoothed RTSTRUCTs where smoothing passed. Please import each
RTSTRUCT separately against that case's exact `dicom/t2` series. The severe
fragmented case `20240903_45fd973241d1ff6efe2e` has a new one-outer-contour
control, but no smoothed file because the smoothing candidate still failed the
surface, hole, and lesion-containment gates. That failure is intentional rather
than silently accepting unsafe geometry.

The files pass our code-level DICOM/reference checks and exact mask round-trip
checks, but I am not calling the UroFusion issue resolved until you confirm the
live imports, alignment, independent ROI toggles, labels, and absence of inner
or connector contours. Each case includes a QC image, manifest, and review CSV.

For onboarding/paperwork, these objects remain AI-generated, `UNAPPROVED`, and
research-only with radiologist review required. The canonical SR/findings
redesign is still a later gated change and was not mixed into this fix.

## CARPL inbound DICOM series-filter requirement

CARPL should provide one complete selected series for each of T2, ADC, and
high-b diffusion from the same MR study. Matching is case-insensitive against
`Modality` and `SeriesDescription`.

### T2

- `Modality = MR`.
- `SeriesDescription` contains `t2` and an axial marker (`tra`, standalone
  `ax`, `axial`, or an `ax_`/`_ax_`-style marker).
- Exclude oblique, coronal, sagittal, pelvis/whole-pelvis, TIRM, STIR, SPACE,
  localizer, survey, T1, post-contrast (`+c`), ADC, DWI/diffusion/TRACEW, and
  fat-suppressed markers.

Current implementation regex:

```text
^(?!.*(?:obl|oblq|oblique|obq|cor|sag|pelvis|whole[_ ]?pelvis|wholepelvis|tirm|stir|space|localizer|survey|t1|\+c|adc|dwi|diff|tracew?|_fs_|_fs\b|\bfs_|\bfs\b))(?=.*t2)(?=.*(?:tra|\bax\b|axial|^ax[_ ]|_ax[_ ]|AX[_ ])).*
```

### ADC

- `Modality = MR`.
- `SeriesDescription` contains `adc`.
- Exclude localizer and survey series.

Current implementation regex:

```text
^(?!.*(?:localizer|survey))(?=.*adc).*
```

### High-b diffusion

- `Modality = MR`.
- Exclude descriptions containing `adc`, `localizer`, or `survey`.
- Include descriptions containing `tracew`, `dwi`, `diff`, `calc`, `computed`,
  or `computer`; also accept `focus`/`muse` when followed by a b-value marker.
- If several b-values share one Series Instance UID, retain only instances at
  the highest numeric b-value. Read standard Diffusion b-value `(0018,9087)`
  first and Siemens private `(0019,100C)` only as a fallback. If neither tag is
  present, do not guess a b-value; flag the series for review.

Current series-selection regex:

```text
^(?!.*adc)(?!.*(?:localizer|survey))(?:(?=.*tracew?)|(?=.*dwi)|(?=.*diff)|(?=.*calc)|(?=.*compute[dr]?)|(?=.*(?:focus|muse).*b[0-9])).*
```

The request for CARPL to flag missing b-value tags is intentionally stricter
than the current legacy fallback, which passes an untagged matched diffusion
series through unfiltered. That fallback has not been changed in this Gate 1
patch; any change to it needs a separate integration test and approval.

### Required image integrity

- Send single-frame MR Image Storage instances, not Raw Data Storage or other
  non-image DICOM objects.
- Preserve original Study Instance UID, Series Instance UID, SOP Instance UID,
  Frame of Reference UID, accession/de-identification attributes, and patient
  attributes exactly, including genuine blank values.
- Each selected series must have a consistent matrix, pixel spacing, image
  orientation, and Frame of Reference; unique SOP Instance UIDs and slice
  positions; and a complete, uniformly spaced spatial stack.
- Every spatial image needs `ImagePositionPatient`,
  `ImageOrientationPatient`, `PixelSpacing`, `Rows`, and `Columns`.
- CARPL should return exactly one selected series per required input or flag an
  ambiguity instead of choosing silently.
- The selected T2 Series Instance UID is the reference series for derived
  RTSTRUCT/SR evidence. Every RTSTRUCT contour must reference the actual T2
  slice SOP Instance UID from that series.

Please ask CARPL to confirm how its filter expresses these rules and how it
reports zero matches, multiple matches, missing b-value tags, or incomplete
spatial stacks. Those cases must fail visibly rather than be guessed.
