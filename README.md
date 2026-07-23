# Prostate MRI Lesion Segmentation and Classification (MONAI Deploy Pipeline)

<p float="left">
  <img src="imgs/workflow-diagram.png" width="900"/>
</p>

This application ingests multi-parametric prostate MRI DICOM studies (T2-weighted, ADC, and high b-value images), segments the prostate organ and lesions, and classifies detected lesions into PI‑RADS categories. It is implemented as a MONAI Deploy DAG with dedicated operators for ingestion, preprocessing, inference, and reporting.

This workflow takes T2, ADC, and HighB MRI series as input and produces several NIfTI files, DICOM RT Structure Sets (RTSTRUCTs), and a classification report as output. These outputs contain organ and lesion segmentations, lesion probability maps, and per-lesion PI‑RADS scores. A combined RTSTRUCT merging organ zones and lesion contours into a single file is also generated for convenient viewing in clinical DICOM viewers.

<p float="left">
  <img src="imgs/anson_organ_seg.png" width="400" height="300" />
  <img src="imgs/anson_all_seg.png" width="400" height="300" />
  <img src="imgs/anson_lesion.png" width="400" height="300" />
</p>

## What goes in (Inputs)

- DICOM study folder(s) containing at least one valid series for each of:
  - T2-weighted (T2)
  - Diffusion-derived ADC map (ADC)
  - High b‑value diffusion (HIGHB)
- Series auto-selection is done via curated regex rules on DICOM tags (Modality, SeriesDescription). The rules were derived from Dr. Anh Tuan Doan's manual annotation of ~3,500 SGH cases and validated at 100 % precision / 100 % recall on 43,470 annotated rows (see `scripts/validate_selection_rules.py`). The rules work with ProstateX-style naming and are robust across common site conventions including SGH de-identified data.
- For multi-b-value HIGHB series (e.g. `ep2d_diff_b50_500_1000_1800`), only the highest b‑value slices are retained automatically via the `HighBValueFilterOperator`.


## What comes out (Outputs)

- NIfTI versions of inputs (for reproducibility and downstream checks)
  - `output/t2/t2.nii.gz`
  - `output/adc/adc.nii.gz`
  - `output/highb/highb.nii.gz`
- Prostate organ segmentation (multi-class: background, TZ, PZ)
  - `output/organ/organ.nii.gz` (original multi-class mask: 0=background, 1=TZ, 2=PZ — kept for reference/review)
  - `output/organ/cleaned_organ.nii.gz` (cleaned mask: erroneous islands removed, holes filled — used for all downstream processing)
  - `output/organ/cleanup_metrics.json` (component counts, removed/fill voxel counts, largest-component fraction)
- Lesion segmentation (per-fold and merged probabilities + final mask)
  - `output/lesion/fold{0..4}_lesion_prob.nii.gz` (5 folds)
  - `output/lesion/merged_lesion_prob.nii.gz`
  - `output/lesion/lesion_mask.nii.gz` (binary mask; no component-size filtering is applied without clinical validation)
- Lesion report with PI‑RADS and lesion statistics
  - `output/lesions.txt` (YAML: per-lesion PI‑RADS, major axis length, volume; prostate volume in mm³/cc and LR/AP/SI dimensions in mm)
  - `output/prostate_measurements_SR.dcm` (DICOM Comprehensive SR / TID 1500 measurement report; COMPLETE, UNVERIFIED)
- RTSTRUCT files (DICOM RT Structure Set), all referenced against the pipeline-selected T2 DICOM series
  - `output/organ/organ_RTSTRUCT.dcm` — original organ zones (kept for reference/review by radiologists)
  - `output/organ/cleaned_organ_RTSTRUCT.dcm` — cleaned organ zones (islands removed, clinically accurate)
  - `output/lesion/lesion_RTSTRUCT.dcm` — lesion contours only (ROI: `Lesion` red)
  - `output/combined_organ_lesion_RTSTRUCT.dcm` — **combined** cleaned organ zones + lesion contours in a single file for convenient loading in clinical viewers (e.g. CARPL)
- Copies of the pipeline-selected DICOM series (for importing alongside RTSTRUCTs into a viewer)
  - `output/dicom/t2/` — T2 DICOM files
  - `output/dicom/adc/` — ADC DICOM files
  - `output/dicom/highb/` — highest-b-value DICOM files only


## End-to-end pipeline (high level)

1) **Data ingestion**
- Load DICOM studies → select T2/ADC/HIGHB series using regex rules on SeriesDescription → convert each series to in‑memory 3D images with NIfTI metadata.
- For multi-b-value HIGHB series, a `HighBValueFilterOperator` retains only the highest b‑value slices before volume conversion.

2) **Organ segmentation (prostate)**
- Preprocess T2: channel-first, RAS orientation, 1.0 mm isotropic resampling, intensity normalization.
- Infer prostate mask with a packaged MONAI model (sliding-window) and invert transforms to the original space.
- Save `output/organ/organ.nii.gz` as a multi-class mask (background/TZ/PZ).

3) **Lesion segmentation (ensemble, organ-masked)**
- **Organ mask cleanup** (new): after saving the original `organ.nii.gz`, produce `cleaned_organ.nii.gz` by removing disconnected islands (keep largest connected component) and filling holes. The cleaned mask is used for all downstream processing; the original is retained for radiologist reference.
- If the organ mask is empty (no prostate detected), lesion inference is skipped and an all-zero `lesion_mask.nii.gz` is written.
- Save T2/ADC/HIGHB/organ as NIfTI for reproducible preprocessing.
- Align ADC/HIGHB to T2 geometry; resample all to 0.5 mm isotropic.
- Compute ROI from the **cleaned** organ mask with a 32‑voxel margin and crop volumes to the ROI.
- Run a 5‑fold 3D RR‑UNet ensemble on the ROI, accumulate and average per-voxel probabilities.
- Parallel inference across all 5 folds using `ThreadPoolExecutor`.
- Multiply merged probabilities by the **cleaned** organ mask (remove out-of-prostate predictions).
- Threshold with a fixed value (0.63) → `lesion_mask.nii.gz`.
- Lesion components are not size-filtered: a safe minimum clinically significant volume has not yet been validated, and multifocal/small lesions must not be removed automatically.

4) **Lesion classification (PI‑RADS)**
- Resample T2/ADC/HIGHB/organ/lesion to 0.5 mm.
- Extract centered 64×64×64 crops around each connected component in the lesion mask.
- Classify crops using a lightweight 3D ResNet; map class index {0..3} → PI‑RADS {2..5}.
- Rule-based adjustment: if predicted=2 (PI‑RADS 4) and major axis length > 40 mm, upgrade to predicted=3 (PI‑RADS 5).
- **Prostate measurements**: compute true mask volume from the NIfTI affine determinant and physical LR/AP/SI bounding extents in RAS coordinates from the cleaned organ mask.
- Write lesion and organ stats to `output/lesions.txt`.
- Write the same prostate measurements to `output/prostate_measurements_SR.dcm`, a standards-based DICOM Comprehensive SR that references the selected T2 MR series. Algorithm outputs are marked COMPLETE but UNVERIFIED and not FINAL.

5) **RTSTRUCT generation and DICOM export**
- Generate four DICOM RT Structure Sets from the NIfTI masks, all referenced against the pipeline-selected T2 DICOM series (matched by `SeriesInstanceUID`):
  - `organ_RTSTRUCT.dcm` — original organ zones (for reference/review).
  - `cleaned_organ_RTSTRUCT.dcm` — cleaned organ zones (islands removed, clinically accurate).
  - `lesion_RTSTRUCT.dcm` — lesion contour(s).
  - `combined_organ_lesion_RTSTRUCT.dcm` — cleaned organ + lesion ROIs in a single file.
- Contours use standards-based `CLOSEDPLANAR_XOR` encoding with pinhole/keyhole cuts disabled. Point- or line-like components are represented as valid sub-pixel polygons instead of being discarded, and contour approximation is disabled so the RTSTRUCT rasterizes exactly back to the exported mask.
- Copy the pipeline-selected DICOM files for T2, ADC, and HIGHB into `output/dicom/` so radiologists can import DICOMs and RTSTRUCTs together into a viewer.


## Detailed components

### Data ingestion and series selection
- Orchestrated by `prostate_mri_lesion_seg_app/app.py` using MONAI Deploy:
  - `DICOMDataLoaderOperator` → `DICOMSeriesSelectorOperator` (rules per modality) → `DICOMSeriesToVolumeOperator` (in‑memory images).
  - For the HIGHB branch, a `HighBValueFilterOperator` is inserted between the selector and the volume converter to retain only the highest b‑value slices.
- Selection rules are regex-based on `SeriesDescription` (case-insensitive). They were derived from Dr. Anh Tuan Doan's manual annotation of ~3,500 SGH de-identified cases and validated at 100 % precision / 100 % recall on 43,470 annotated rows via `scripts/validate_selection_rules.py`.
- `ImageType` is intentionally not required in the rules because SGH data frequently lacks this field; the regex alone is strict enough to separate T2 / ADC / HIGHB.
- The DAG wires T2/ADC/HIGHB into downstream operators (organ seg → lesion seg → classifier).
- A monkeypatch on `DICOMSeriesToVolumeOperator.prepare_series` fixes two bugs with SGH de-identified DICOM data: (1) incorrect slice removal using `enumerate()` indices instead of actual indices, and (2) crash on missing `.distance` attributes after broken removal.

### High b‑value filtering
- Operator: `HighBValueFilterOperator` (`prostate_mri_lesion_seg_app/highb_filter_operator.py`)
- Problem: SGH multi-b DWI series (e.g. `ep2d_diff_b50_500_1000_1800 prostate_TRACEW`) pack all b-value volumes into a single DICOM series. A 24-slice acquisition at 4 b-values produces 96 DICOM instances. Without filtering, `DICOMSeriesToVolumeOperator` stacks all instances into one 3-D volume whose slices alternate between b-values, producing a physically meaningless input for the lesion model.
- Solution: reads each SOP instance's b-value (standard tag `(0018,9087)` first, Siemens private tag `(0019,100C)` as fallback), identifies the maximum b-value, and removes all instances that do not carry the maximum.
- B-values are rounded to the nearest integer before comparison to handle floating-point storage (e.g. 1799.99 → 1800).
- If no b-value tags are found (e.g. some GE sequences), the series passes through unmodified (safe default).

### Organ segmentation
- Operator: `ProstateSegOperator` (`prostate_mri_lesion_seg_app/organ_seg_operator.py`)
- Preprocessing (T2):
  - Channel-first, RAS orientation, `Spacingd` to 1.0 mm, channel-wise z‑score normalization.
- Inference:
  - `MonaiSegInferenceOperator`, sliding-window ROI `(128, 128, 16)`, 50% overlap, batch size 4.
  - Model artifact: `models/organ/model.ts` (TorchScript).
  - Model loading: Operator looks for `model.ts` directly in the `model_path` directory.
- Post-processing:
  - Softmax → invert transforms → `AsDiscreted(argmax=True)` → multi-class prostate mask (background/TZ/PZ) in original space.
  - The `argmax=True` parameter is critical to preserve all three classes (0=background, 1=TZ, 2=PZ); using only a threshold without argmax would collapse to binary.

### Lesion segmentation (3D RR‑UNet ensemble, 5 folds)
- Operator: `ProstateLesionSegOperator` (`prostate_mri_lesion_seg_app/custom_lesion_seg_operator.py`)
- Inputs: T2, ADC, HIGHB, and organ mask (from Organ Seg).
- **Empty organ guard**: if the organ mask is entirely zero (no prostate detected — common with very-thin SGH series of 1–6 slices), lesion inference is skipped entirely and an all-zero `lesion_mask.nii.gz` is written. This avoids meaningless full-volume inference.
- Preprocessing:
  - Save all inputs as NIfTI under `output/`.
  - Graceful affine fallback: `nifti_affine_transform` → `dicom_affine_transform` → identity matrix (handles SGH data with missing spatial metadata).
  - Resample ADC/HIGHB to match T2 geometry (SimpleITK), then resample all to 0.5 mm isotropic.
  - Compute a prostate-centered ROI using `organ.nii.gz`, with a 32‑voxel margin; crop the inputs to the ROI.
  - Per-channel z‑score normalization across `[T2, ADC, HIGHB]`.
- Model:
  - `RRUNet3D` (residual encoder–decoder), 3 input channels, 2-class softmax output.
  - 5 instances with weights from `models/fold{0..4}/model_best_fold*.pth.tar`.
- Inference:
  - Patch-based over the ROI with strides sized to multiples of 32; accumulate sums and counts per voxel, then average.
  - Reconstruct to original size and save per-fold probability maps; average to `merged_lesion_prob.nii.gz`.
  - Parallel inference across all 5 folds using `ThreadPoolExecutor`.
- Post-processing:
  - Multiply merged probabilities by the cleaned organ mask (removes non-prostate detections).
  - Fixed threshold (0.63) → `lesion_mask.nii.gz`.
- RTSTRUCT and DICOM export (runs after lesion inference):
  - Generates four RTSTRUCT files via `rtstruct_utils.generate_rtstruct_files()` — original organ, cleaned organ, lesion, and combined (see Outputs above).
  - RTSTRUCTs are built from the exact T2 DICOM series the pipeline selected, matched by `SeriesInstanceUID` (not heuristic folder-name matching).
  - Separate `CLOSEDPLANAR_XOR` contours preserve inner/outer boundaries without the visible connector lines produced by `rt-utils` pinhole encoding. Degenerate OpenCV contours are repaired into valid pixel-footprint polygons and round-trip checked by the validation harness.
  - SGH de-identified data often lacks the `StudyID` tag `(0020,0010)`, which crashes `rt_utils`. The code automatically patches `StudyID` from `AccessionNumber` in a temporary copy of the DICOM files.
  - Copies the pipeline-selected T2, ADC, and HIGHB DICOM series into `output/dicom/{t2,adc,highb}/` so radiologists can import DICOMs + RTSTRUCTs together into a viewer. For HIGHB, only the highest b‑value files are copied.

### Lesion classification (PI‑RADS)
- Operator: `ProstateLesionClassifierOperator` (`prostate_mri_lesion_seg_app/custom_lesion_classifier_operator.py`)
- Inputs: T2, ADC, HIGHB, organ mask, lesion mask.
- Preprocessing:
  - `ResampleToMatch` ADC/HIGHB to T2, then resample all channels (and masks) to 0.5 mm isotropic.
  - Combine channels, normalize the first 3 (T2/ADC/HIGHB).
  - Extract 64×64×64 crops centered on each 3D lesion component.
- Model:
  - 3D ResNet (`resnet.py`), BasicBlock, layers `[1,1,1,1]`, channels `[32,64,128,256]`, `n_classes=4`.
  - Weights at `models/classifier/model_best.pth.tar`.
- Inference and rules:
  - Predict class index ∈ {0..3}; PI‑RADS = index + 2.
  - If predicted=2 (PI‑RADS 4) and major axis > 40 mm → promote to PI‑RADS 5.
- Reporting:
  - `lesions.txt` includes per-lesion ID, major axis length (mm), volume (mm³), PI‑RADS, plus cleaned-prostate volume and LR/AP/SI dimensions.
  - `prostate_measurements_SR.dcm` encodes prostate dimensions and volume as a DICOM TID 1500 Imaging Measurement Report using UCUM units. It references the selected T2 evidence and is explicitly UNVERIFIED.


## How to run

### Software and Setup

In order to run this workflow and build a MAP you will need to [install MONAI Deploy App SDK](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/getting_started/installing_app_sdk.html). This can be installed along with all other software dependencies by running `pip install -r prostate_mri_lesion_seg_app/requirements.txt`.

It is recommended to have an NVIDIA GPU with at least 12 GB of memory available.

### Models

The models needed to build and execute the pipeline (1 organ segmentation model, 5 lesion segmentation models, 1 classification model) are hosted separately on [Google Drive here](https://drive.google.com/drive/folders/1EpjrlzEdV7CcaCYqGTIEzOapamP4Ag6M?usp=sharing).

Download these models and put them inside a folder named `prostate_mri_lesion_seg_app/models` alongside the rest of the application code. Pipeline creation and execution will not complete if the model file path is changed or renamed.

The expected model directory structure:
```
models/
├── organ/
│   └── model.ts                    # TorchScript organ segmentation model
├── fold0/
│   └── model_best_fold0.pth.tar   # Lesion segmentation fold 0
├── fold1/
│   └── model_best_fold1.pth.tar   # Lesion segmentation fold 1
├── fold2/
│   └── model_best_fold2.pth.tar   # Lesion segmentation fold 2
├── fold3/
│   └── model_best_fold3.pth.tar   # Lesion segmentation fold 3
├── fold4/
│   └── model_best_fold4.pth.tar   # Lesion segmentation fold 4
└── classifier/
    └── model_best.pth.tar          # PI-RADS classification model
```

### Local (non-containerized)

Requirements are enumerated in `prostate_mri_lesion_seg_app/requirements.txt`. A quick way to run end‑to‑end:

```bash
./scripts/test_local.sh -i test-data/ProstateX-0004/ -o output/ -m models/
```

- GPU is used by default. Add `-c` to force CPU:

```bash
./scripts/test_local.sh -i test-data/ProstateX-0004/ -o output/ -m models/ -c
```

### MAP (containerized via Holoscan CLI)

Build and run a MONAI Application Package (requires Holoscan CLI):

```bash
# Build (optional flag -b in helper script)
./scripts/test_MAP.sh -i test-data/ProstateX-0004/ -o output/ -m prostate_mri_lesion_seg_app/models/ -b

# Run a prebuilt MAP
./scripts/test_MAP.sh -i test-data/ProstateX-0004/ -o output/ -m prostate_mri_lesion_seg_app/models/
```

### Docker (containerized, for HIVE / air-gapped deployment)

A standalone Docker image is provided under `docker/` for deployment on machines without the full MONAI Deploy SDK (e.g. HIVE). The image bundles only the application code and Python dependencies; models, input data, and output are volume-mounted at runtime.

**Build and save the Docker image:**

```bash
python scripts/build_and_save.py
# Produces docker/docker_image.tar (~4–6 GB)
```

Docker images are immutable snapshots of the application code. After changing
the pipeline, rebuild `docker/docker_image.tar`, transfer that new tar to the
target machine/HIVE, and load it before inference. An older already-loaded
image — including one selected with `--skip-load` — will continue to generate
the older outputs.

**Load and run a single patient:**

```bash
python scripts/load_and_run.py -i test-data/ProstateX-0004/ -o output/ -m models/
# Or with a pre-saved tar:
python scripts/load_and_run.py -i <input_dir> -o <output_dir> -m <models_dir> --tar docker/docker_image.tar
# CPU mode (much slower):
python scripts/load_and_run.py -i <input_dir> -o <output_dir> -m <models_dir> --cpu
```

**Batch-process multiple patients (Docker):**

```bash
python scripts/batch_load_and_run.py \
    -i /data/patients/ -o /output/ -m /models/ \
    --tar docker/docker_image.tar \
    -r --cooldown 5
```

**Docker Compose** (single patient, convenience):

```bash
INPUT_DIR=/data/patient01 OUTPUT_DIR=/output/patient01 MODEL_DIR=/models \
    docker compose -f docker/docker-compose.yaml up
```

The Docker image uses NVIDIA CUDA 12.4 + cuDNN runtime (Ubuntu 22.04, Python 3.10) with PyTorch 2.6.0 (CUDA 12.4).

### Batch processing (local, non-Docker)

```bash
./scripts/test_batch.sh -i /data/patients/ -o /output/ -m models/
# With resume and cooldown:
./scripts/test_batch.sh -i /data/patients/ -o /output/ -m models/ -r --cooldown 30
```

Local batch runs execute the application directly from
`prostate_mri_lesion_seg_app/`, so every newly processed case uses the current
source code. Existing outputs are not modified retroactively.

To verify a pipeline revision, use a new output directory without `-r`:

```bash
bash ./scripts/test_batch.sh \
    -i test-data/ \
    -o output_golf/ \
    -m models/ \
    --cooldown 10
```

Running without `-r` is a fresh batch and clears the selected output root
before processing. Do not point it at results that must be retained.

Batch processing features:
- **Resume mode** (`-r`): skips complete cases based on required artifact presence; incomplete/legacy outputs are cleared and rerun. Resume cannot identify an internal format revision when all expected files already exist. For example, a complete case containing an older keyhole RTSTRUCT will be skipped rather than regenerated with `CLOSEDPLANAR_XOR`. Use a new output root, run a fresh batch without `-r`, or remove the specific case directory before resuming.
- **Cooldown** (`--cooldown SECS`): pauses between cases to prevent thermal issues (default 30 s).
- **Single-writer lock**: rejects a second batch targeting the same output root, preventing output/audit corruption.
- **Audit CSV** (`--audit-csv PATH`): writes status, missing modalities, cleanup metrics, elapsed time, and file-presence flags (default `<output>/batch_audit_log.csv`).
- **Final guardrail exit**: the batch exits non-zero if any audit status is `FAIL_*`, even when every underlying MONAI invocation returned exit code 0.
- Status vocabulary: `SUCCESS_COMPLETE`, `SUCCESS_EMPTY_LESION`, `FAIL_MISSING_MODALITY`, `FAIL_ORGAN_SEG`, `FAIL_POSTPROCESSING`, `FAIL_LESION_SEG`, `FAIL_CLASSIFIER`, `FAIL_REPORTING`, `FAIL_RTSTRUCT`, `FAIL_RUNTIME`, `FAIL_EMPTY_OUTPUT`, `SKIPPED_RESUME`.

In summary:
- A new local inference run produces the latest outputs, including no-keyhole `CLOSEDPLANAR_XOR` RTSTRUCTs.
- A Docker/HIVE inference run produces them only after rebuilding and loading the updated image.
- A complete case skipped by `-r` keeps its existing files and encoding.

### Using this Repository

The easiest way to get started with this workflow is to run a test image taken from the [ProstateX](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=23691656) dataset. The `build_and_run.ipynb` file walks through this process using ProstateX-0004 which can be [downloaded separately](https://drive.google.com/drive/folders/1besSncSLlbeiv7UWveRJoOYQXOzu3JkU?usp=sharing) and placed in a `test-data/` directory.

After validating on test data, you can test this image on your own study or dataset. One of the main considerations when adapting to a new dataset will be making sure the [DICOM Series Selector Operator](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/modules/_autosummary/monai.deploy.operators.DICOMSeriesSelectorOperator.html#monai.deploy.operators.DICOMSeriesSelectorOperator) is configured to properly differentiate between the different naming schemes and properties of the new dataset.

If all three (T2, ADC, HighB) series are not detected properly in the study, the pipeline will not complete. If any of these modalities are incorrectly routed, the pipeline results will not be accurate. The workflow currently saves intermediate copies (in NIfTI) of these series in the output folder so it is possible to verify they were picked up (and preprocessed) correctly.

The current set of rules in `app.py` filter based on SeriesDescription and work with both ProstateX and SGH de-identified datasets. They were validated on ~43,500 annotated rows — see `scripts/validate_selection_rules.py` for the validation harness. Please refer to MONAI documentation for guidance on modifying these rules for custom filtering.


## Training

The repository includes training infrastructure for both organ and lesion segmentation models.

### Organ Segmentation Training

- Notebook: `notebooks/train_organ.ipynb`
- Model: Multi-class RRUNet3D (3 output channels: background, TZ, PZ)
- Training script: `training/train_organ.py`
- Configuration: `configs/organ.yaml`
- 5-fold cross-validation training
- Output: TorchScript model (`model.ts`) for deployment

### Lesion Segmentation Training

- Notebook: `notebooks/train_lesion.ipynb`
- Model: Binary RRUNet3D (2 output channels: background, lesion)
- Training script: `training/train_lesion.py`
- Configuration: `configs/lesion.yaml`
- 5-fold cross-validation training (ensemble)
- Input: Multi-parametric (T2 + ADC + HIGHB, 3 channels)
- ROI cropping with 32-voxel margin
- Handles missing lesion masks (creates empty masks for patients without lesions)

### Training Infrastructure

- Training engine: `training/engine.py`
- Datasets: `training/datasets.py`
- Transforms: `training/transforms.py`
- Losses: `training/losses.py`
- Metrics: `training/metrics.py`
- Utilities: `training/utils.py`


## Evaluation

- Compare organ and lesion masks using DICE:

```bash
./scripts/compare_output.sh <reference_dir> <output_dir>
```

This calls `scripts/eval_dice.py` under the hood (loads two NIfTI files, computes mean DICE).


## Models and artifacts

- Organ segmentation: `models/organ/model.ts` (TorchScript, used by MONAI Deploy inference operator).
- Lesion segmentation: `models/fold0..fold4/model_best_fold*.pth.tar` (5‑fold RR‑UNet ensemble).
- Lesion classifier: `models/classifier/model_best.pth.tar` (3D ResNet, 4 classes → PI‑RADS 2–5).
- Outputs are written under `output/` as described above (NIfTI volumes, RTSTRUCTs, DICOM copies, and classification report).


## Assumptions and constraints

- Assumes valid T2, ADC, and high b‑value series are present; selection is regex-based, tuned for ProstateX and SGH de-identified naming conventions.
- For multi-b-value HIGHB series, only the highest b‑value slices are used; lower b‑value slices are discarded by `HighBValueFilterOperator`.
- Resampling strategy: organ seg at 1.0 mm; lesion seg/classifier at 0.5 mm isotropic.
- Lesion segmentation is explicitly organ‑constrained:
  - ROI crop for inference is centered on the prostate mask with a margin.
  - Post-inference probabilities are multiplied by the organ mask prior to thresholding.
  - If the organ mask is empty (no prostate detected), lesion inference is skipped entirely.
- Fixed lesion threshold (0.63) after ensemble averaging.
- Organ segmentation produces multi-class output (background/TZ/PZ) via argmax post-processing.
- RTSTRUCT generation requires the `rt-utils` package and handles SGH de-identified data quirks (missing `StudyID` tag, slice count mismatches between NIfTI and DICOM).
- GPU strongly recommended (PyTorch + 3D inference); CPU fallback supported.


## Comparison-ready highlights

- Multi-parametric inputs (T2 + ADC + high b‑value), with curated series selection rules validated on ~43,500 annotated rows, increase robustness to site naming variability.
- Automatic highest-b-value filtering handles multi-b DWI series without manual intervention.
- Two-stage approach:
  - Organ segmentation ensures prostate-focused processing and provides multi-class anatomical zones (TZ/PZ).
  - Lesion segmentation is both ROI‑focused and organ‑masked for fewer false positives.
- 5‑fold ensemble improves lesion robustness and calibration; fold probabilities are exported for transparency.
- PI‑RADS classification uses 3D crops and a size-based rule to better separate 4 vs. 5.
- Clear, audit-friendly artifacts: NIfTI inputs/outputs, cleanup metrics, merged probabilities, final masks, four RTSTRUCT files, copied DICOM series, YAML report, and DICOM SR.
- Combined RTSTRUCT merges organ zones and lesion contours into a single DICOM RT Structure Set for convenient viewing in clinical DICOM viewers (e.g. CARPL).
- Batch pipeline includes per-case audit CSV with structured status classification, enabling systematic quality review over large cohorts.


## Key source files

- Pipeline and DAG: `prostate_mri_lesion_seg_app/app.py`
- Organ segmentation operator: `prostate_mri_lesion_seg_app/organ_seg_operator.py`
- Lesion segmentation operator: `prostate_mri_lesion_seg_app/custom_lesion_seg_operator.py`
- Lesion classifier operator: `prostate_mri_lesion_seg_app/custom_lesion_classifier_operator.py`
- High b‑value filter operator: `prostate_mri_lesion_seg_app/highb_filter_operator.py`
- Models: `prostate_mri_lesion_seg_app/rrunet3D.py`, `prostate_mri_lesion_seg_app/resnet.py`
- Utilities: `prostate_mri_lesion_seg_app/common.py`, `prostate_mri_lesion_seg_app/rtstruct_utils.py`
- DICOM SR export: `prostate_mri_lesion_seg_app/dicom_sr_utils.py`
- Training: `training/engine.py`, `training/train_organ.py`, `training/train_lesion.py`
- Docker: `docker/Dockerfile`, `docker/docker-compose.yaml`, `docker/requirements.txt`
- Scripts: see [Scripts](#scripts) section below


## Quick sanity check after a run

- Confirm outputs exist:
  - `output/t2/t2.nii.gz`, `output/adc/adc.nii.gz`, `output/highb/highb.nii.gz`
  - `output/organ/organ.nii.gz` (original, multi-class: 0=background, 1=TZ, 2=PZ)
  - `output/organ/cleaned_organ.nii.gz` (cleaned: islands removed, holes filled)
  - `output/organ/cleanup_metrics.json` (cleanup audit)
  - `output/lesion/fold*_lesion_prob.nii.gz`, `output/lesion/merged_lesion_prob.nii.gz`, `output/lesion/lesion_mask.nii.gz`
  - `output/lesions.txt` (PI‑RADS per lesion, prostate volume in mm³/cc, 3 orthogonal dimensions)
  - `output/prostate_measurements_SR.dcm` (DICOM SR, COMPLETE/UNVERIFIED)
  - `output/organ/organ_RTSTRUCT.dcm` (original), `output/organ/cleaned_organ_RTSTRUCT.dcm` (cleaned)
  - `output/lesion/lesion_RTSTRUCT.dcm`, `output/combined_organ_lesion_RTSTRUCT.dcm` (uses cleaned organ)
  - `output/dicom/t2/`, `output/dicom/adc/`, `output/dicom/highb/` (copies of pipeline-selected DICOM series)
- For batch runs, inspect the audit CSV (`batch_audit_log.csv`) for per-case status:
  ```bash
  python scripts/case_status.py summary output/batch_audit_log.csv
  ```

If you need a simple visualization, see `build_and_run.ipynb` for an example to overlay organ/lesion masks on T2.


## Scripts

There are several scripts to help with running, validation, deployment, and development included in the `scripts/` directory.

### Running

- `scripts/test_local.sh`: Execute workflow locally on test images without building a MAP.
- `scripts/test_MAP.sh`: Execute MAP workflow on test images with option to rebuild MAP.
- `scripts/test_batch.sh`: Batch processing script for multiple patients (local, non-Docker). Supports resume (`-r`), cooldown between cases (`--cooldown`), and per-case audit CSV logging (`--audit-csv`).
- `scripts/load_and_run.py`: Load the Docker image from a `.tar` file and run the pipeline on a single patient inside a container. Used for HIVE / air-gapped deployment.
- `scripts/batch_load_and_run.py`: Batch-process multiple patients through the Dockerised pipeline. Each patient runs in a fresh container for crash isolation and clean GPU memory. Supports resume, cooldown, and audit CSV (same format as `test_batch.sh`).

### Docker build

- `scripts/build_and_save.py`: Build the `prostate-mri-seg` Docker image from `docker/Dockerfile` and save it as `docker/docker_image.tar` for transfer to HIVE or air-gapped machines.

### Evaluation and validation

- `scripts/compare_output.sh`: Computes organ and lesion DICE scores for two output directories.
- `scripts/eval_dice.py`: Loads two NIfTI files and computes mean DICE (called by `compare_output.sh`).
- `scripts/compare_batch_outputs.py`: Produces a per-case CSV and JSON regression summary for a new batch versus a prior output. It checks input/original-organ identity, quantifies cleaned-organ and lesion changes, and validates each generated DICOM SR against its T2 study.
- `scripts/validate_rtstruct_outputs.py`: Strictly validates all four RTSTRUCTs per case: SOP/study/T2 references, ROI names, `CLOSEDPLANAR_XOR` use, minimum polygon geometry, DICOM short-string limits, and exact RTSTRUCT-to-NIfTI voxel round trips.
- `scripts/validate_selection_rules.py`: Validates T2/ADC/HIGHB selection rules (from `app.py`) against an annotated ground-truth CSV. Reports precision, recall, F1, and lists false positives / negatives per class.
- `tests/`: standard-library `unittest` guardrails for connected-component cleanup, hole filling, affine-aware measurements, DICOM SR parsing, exact no-keyhole RTSTRUCT round trips, and audit status classification:
  ```bash
  python -m unittest discover -s tests -v
  python scripts/validate_rtstruct_outputs.py output/
  ```

### Data investigation

- `scripts/fill_series_description_template.py`: Scan DICOM directories and generate a CSV with per-series metadata (SeriesDescription, b-values, slice counts) and selection rule match flags. Supports multi-threaded processing, resume from interrupted runs, and exhaustive b-value enumeration across all slices.
- `scripts/case_status.py`: Per-case status detection and CSV audit logging module. Inspects output directories for expected files and classifies outcomes into a fixed vocabulary (`SUCCESS_COMPLETE`, `FAIL_MISSING_MODALITY`, etc.). Used by both `test_batch.sh` and `batch_load_and_run.py`.


## Publications

Several publications have leveraged this workflow either in part or in full.

Esengur, Omer Tarik, et al. ["Assessing the Impact of Transition and Peripheral Zone PSA Densities Over Whole‐Gland PSA Density for Prostate Cancer Detection on Multiparametric MRI."](https://onlinelibrary.wiley.com/doi/abs/10.1002/pros.24863) The Prostate (2025): e24863.

Yilmaz, Enis C., et al. ["External Validation of a Previously Developed Deep Learning–based Prostate Lesion Detection Algorithm on Paired External and In-House Biparametric MRI Scans."](https://pubs.rsna.org/doi/abs/10.1148/rycan.240050) Radiology: Imaging Cancer 6.6 (2024): e240050.

Lin, Yue, et al. ["Evaluation of a Cascaded deep learning–based algorithm for prostate lesion detection at biparametric MRI."](https://pubs.rsna.org/doi/abs/10.1148/radiol.230750) Radiology 311.2 (2024): e230750.

Simon, Benjamin D., et al. ["Automated detection and grading of extraprostatic extension of prostate cancer at MRI via cascaded deep learning and random forest classification."](https://www.sciencedirect.com/science/article/abs/pii/S1076633224002204) Academic Radiology 31.10 (2024): 4096-4106.

Johnson, Latrice A., et al. ["Automated prostate gland segmentation in challenging clinical cases: comparison of three artificial intelligence methods."](https://link.springer.com/article/10.1007/s00261-024-04242-7) Abdominal Radiology 49.5 (2024): 1545-1556.


## License

This work was developed by NVIDIA and the NIH National Cancer Institute (NCI). Please refer to the LICENSE for terms of use.
