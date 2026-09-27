# UroFusion RTSTRUCT acceptance and implementation record

Date recorded: 2026-09-27

## External test conclusion

- Dr Anh reported that the vendor successfully loaded the revised RTSTRUCTs in
  UroFusion and displayed the prostate and lesion contours with good
  compatibility.
- The tested deterministic ROI order is retained: ROI 1 is `Prostate`; ROI
  2…N are the separate lesion components.
- Dr Anh found no meaningful anatomical improvement from the smoothed prostate
  comparison. Prostate smoothing is therefore not part of inference output.

This is compatibility acceptance of the file/profile behavior. It does not
change the DICOM `UNAPPROVED` status into a formal radiologist approval of an
individual patient's model segmentation.

## Implemented canonical behavior

- `combined_organ_lesion_RTSTRUCT.dcm` is generated automatically after lesion
  classification; no review flag is required.
- The prostate is a deterministic single-outer-contour representation derived
  from `organ/cleaned_organ.nii.gz` with one prostate contour per occupied T2
  slice.
- Lesion masks are unchanged native 26-connected components from
  `lesion/lesion_mask.nii.gz`.
- Lesions are matched to classifier scores only by reciprocal physical overlap
  and named `Lesion N - AI PI-RADS X`; ambiguous mappings say `unavailable`.
- The canonical file uses Explicit VR Little Endian, `CLOSED_PLANAR`,
  `AUTOMATIC`, prostate `ORGAN`, lesion `GTV`, and `UNAPPROVED`.
- Every contour references its actual source T2 SOP Instance UID. Generation
  rejects grid/reference mismatches and validates exact mask round trips before
  atomic publication.

## One-lesion versus two-lesion viewer presentation

The tested one- and two-lesion RTSTRUCTs both use the same flat DICOM structure.
There is no nested lesion sequence. UroFusion visually creates a `Lesion`
group when several similarly named lesion ROIs are present. No empty or dummy
parent ROI is added because that would create a misleading extra structure.

## Retired review behavior

- No smoothed RTSTRUCT or smoothed NIfTI is produced.
- The `--urofusion-review` wrapper option, container environment flag, resume
  backfill, standalone review generator, QC/reviewer CSV generation, and
  smoothing implementation have been removed from maintained inference code.
- The August handoff is preserved unchanged as the evidence used for the live
  test; it is not part of new inference output.

## Resume and validation guardrails

Resume now checks the canonical `AI_PROSTATE` profile rather than accepting a
file based on its filename alone. A complete older case with a legacy combined
RTSTRUCT is rerun instead of silently skipped. Automated tests and
`scripts/validate_rtstruct_outputs.py` remain as regression safeguards.
