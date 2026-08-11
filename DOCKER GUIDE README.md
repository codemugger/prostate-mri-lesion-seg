# Docker Guide – Prostate MRI Lesion Segmentation

Transferring the segmentation pipeline from your development laptop to **HIVE** (or any air-gapped / high-compute machine) using Docker.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────┐
│  Docker image  (prostate-mri-seg:latest)             │
│  ┌──────────────────────────────────────────────────┐ │
│  │  NVIDIA CUDA 12.4 + cuDNN   (Ubuntu 22.04)      │ │
│  │  Python 3.10 + PyTorch 2.6.0 (CUDA)             │ │
│  │  MONAI Deploy + Holoscan + all dependencies      │ │
│  │  prostate_mri_lesion_seg_app/  (application code)│ │
│  └──────────────────────────────────────────────────┘ │
│                                                      │
│  Volume mounts (live OUTSIDE the container):         │
│    /input   ← DICOM study folder      (read-only)   │
│    /output  ← pipeline results        (read-write)  │
│    /models  ← model weight files      (read-only)   │
└──────────────────────────────────────────────────────┘
```

Models, input data, and output are **volume-mounted** – they are never
baked into the Docker image. This keeps the image portable and allows you
to swap data / models without rebuilding.

---

## Prerequisites

### On the build machine (your laptop)

| Requirement          | Check command          |
|----------------------|------------------------|
| Docker Engine        | `docker --version`     |
| Python 3.10+         | `python --version`     |
| Internet access      | (to pull CUDA base image & pip packages) |

### On HIVE (the target machine)

| Requirement                 | Check command                |
|-----------------------------|------------------------------|
| Docker Engine               | `docker --version`           |
| NVIDIA GPU (≥ 12 GB VRAM)  | `nvidia-smi`                 |
| NVIDIA Container Toolkit    | `nvidia-ctk --version`       |
| Python 3.10+                | `python --version`           |

> **NVIDIA Container Toolkit** is required for `--gpus all` to work inside
> Docker.  Install guide:
> <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

---

## Step 1 – Build and save the Docker image (on your laptop)

```bash
python scripts/build_and_save.py
```

This will:
1. Build the Docker image `prostate-mri-seg:latest` from `docker/Dockerfile`.
2. Save it to `docker/docker_image.tar`.

The first build downloads the CUDA base image (~4 GB) and installs PyTorch +
all Python dependencies, so expect **10–20 minutes** on a decent connection.
Subsequent builds use the Docker layer cache and finish in seconds (unless
dependencies change).

**Expected output:**
```
Image saved: .../docker/docker_image.tar
File size  : X,XXX,XXX,XXX bytes (X.XX GB)
```

---

## Step 2 – Transfer to HIVE

Copy **four things** to HIVE:

| Item | Source (laptop) | Example HIVE path |
|------|-----------------|-------------------|
| Docker image | `docker/docker_image.tar` | `/hive/prostate-seg/docker_image.tar` |
| Model weights | `prostate_mri_lesion_seg_app/models/` | `/hive/prostate-seg/models/` |
| Input DICOM data | `test-data/ProstateX-0004/` | `/hive/prostate-seg/input/ProstateX-0004/` |
| Run script (single) | `scripts/load_and_run.py` | `/hive/prostate-seg/load_and_run.py` |
| Run script (batch) | `scripts/batch_load_and_run.py` | `/hive/prostate-seg/batch_load_and_run.py` |

### Expected model directory structure
```
models/
├── organ/
│   └── model.ts                    # TorchScript organ segmentation
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
    └── model_best.pth.tar          # PI-RADS classification
```

The run script validates that all 7 model files are present before starting.

---

## Step 3 – Load and run on HIVE

### Option A: Using the helper script (recommended)

```bash
python load_and_run.py \
    -i /hive/prostate-seg/input/ProstateX-0004 \
    -o /hive/prostate-seg/output \
    -m /hive/prostate-seg/models \
    --tar /hive/prostate-seg/docker_image.tar
```

The script will:
1. Validate prerequisites (Docker, GPU, model files).
2. Load `docker_image.tar` into Docker.
3. Run the container with the correct volume mounts.
4. Verify expected output files after completion.

**Useful flags:**
| Flag | Description |
|------|-------------|
| `--skip-load` | Skip `docker load` if the image is already loaded |
| `--cpu` | Run without GPU (much slower, fallback only) |
| `--check-only` | Run prerequisite checks without starting the pipeline |

### Option B: Manual Docker commands

```bash
# 1. Load the image
docker load -i /hive/prostate-seg/docker_image.tar

# 2. Run the pipeline
docker run --rm --gpus all --shm-size=1g \
    -v /hive/prostate-seg/input/ProstateX-0004:/input:ro \
    -v /hive/prostate-seg/output:/output \
    -v /hive/prostate-seg/models:/models:ro \
    prostate-mri-seg:latest
```

### Option C: Docker Compose

```bash
INPUT_DIR=/hive/prostate-seg/input/ProstateX-0004 \
OUTPUT_DIR=/hive/prostate-seg/output              \
MODEL_DIR=/hive/prostate-seg/models               \
docker compose -f docker/docker-compose.yaml up
```

---

## Step 4 – Batch processing (multiple patients)

To process an entire cohort (mirroring `test_batch.sh`), use the batch
wrapper. It loops over every patient subdirectory under `--input`, runs a
**fresh Docker container per patient** (crash-isolated, clean GPU memory),
and supports resume + cooldown – identical behaviour to what you run locally.

```bash
python batch_load_and_run.py \
    -i /hive/data/patients/ \
    -o /hive/output/ \
    -m /hive/models \
    --tar /hive/docker_image.tar \
    -r --cooldown 5
```

| Flag | Description |
|------|-------------|
| `-r` / `--resume` | Skip only complete current-format cases; rerun incomplete or legacy outputs |
| `--cooldown N` | Seconds to pause between patients (default 30, prevents thermal issues) |
| `--skip-load` | Skip `docker load` if image is already loaded |
| `--cpu` | Run without GPU (fallback) |

**Expected output structure** (identical to `test_batch.sh`):
```
output/
├── ProstateX-0004/
│   ├── t2/t2.nii.gz
│   ├── organ/organ.nii.gz
│   ├── organ/cleaned_organ.nii.gz
│   ├── lesion/lesion_mask.nii.gz
│   ├── lesions.txt
│   ├── prostate_measurements_SR.dcm
│   └── ...
├── ProstateX-0010/
│   └── ...
└── PatientN/
    └── ...
```

After completion a summary is printed:
```
Total: 50  Container exit 0: 50  Container non-zero: 0  Resume-skipped: 0
FAIL_EMPTY_OUTPUT          2
SUCCESS_COMPLETE          48
Guardrail result: FAILED (2 case audit failures)
```
The batch returns non-zero if any final audit status begins with `FAIL_`,
including silent failures where the MONAI app itself returned exit code 0.

Both local and Docker batch runners hold a non-blocking lock beside the
output root. A second process targeting the same output directory exits
immediately instead of concurrently modifying cases or the audit CSV.

---

## Per-case audit CSV (silent-failure detection)

Because the MONAI pipeline can exit with code 0 even when a required
modality (T2 / ADC / HIGHB) was rejected by the series selector — or
when the organ model produces an empty mask — the exit-code counters
above are **not** sufficient on their own. Every batch run therefore
writes an audit CSV (one row per case):

```
<output_dir>/batch_audit_log.csv
```

(Override with `--audit-csv <path>` if you want it stored elsewhere.)

### Columns

| Column | Meaning |
|---|---|
| `timestamp_utc` | When the row was written (ISO 8601 UTC) |
| `patient_id` | Case folder name |
| `status` | See status vocabulary below |
| `reason` | Human-readable explanation |
| `missing_modalities` | `;`-separated list of `t2`, `adc`, `highb` missing from output |
| `exit_code` | Docker container exit code (blank on resume-skip) |
| `elapsed_seconds` | Wall time for this case |
| `resume_skipped` | `true` if the case was skipped by resume mode |
| `has_t2_nii`, `has_adc_nii`, `has_highb_nii` | Intermediate NIfTI presence |
| `has_organ_nii`, `has_cleaned_organ_nii` | Original and cleaned organ masks |
| `has_cleanup_metrics` | Organ cleanup audit JSON |
| `has_organ_rtstruct`, `has_cleaned_organ_rtstruct` | Original and cleaned organ RTSTRUCTs |
| `has_lesion_mask`, `has_lesion_rtstruct` | Lesion segmentation outputs |
| `has_merged_lesion_prob`, `has_lesion_folds` | Real lesion inference (vs empty-mask fallback) |
| `has_lesions_txt` | Classifier (PI-RADS) report |
| `has_prostate_sr` | DICOM prostate measurement Structured Report |
| `has_combined_rtstruct` | Combined cleaned-organ + lesion RTSTRUCT |
| `organ_cleanup_status` | `UNCHANGED`, `CLEANED`, `EMPTY`, or `INVALID` |
| `organ_island_count_before`, `organ_removed_island_count`, `organ_removed_voxels`, `organ_filled_hole_voxels`, `organ_largest_component_fraction` | Organ cleanup metrics |
| `output_file_count` | Total files under the case output dir |
| `input_dir`, `output_dir` | Absolute paths |

### Status vocabulary

| `status` | Meaning | Action |
|---|---|---|
| `SUCCESS_COMPLETE` | Organ cleanup, lesion inference, RTSTRUCTs, YAML report, and DICOM SR all completed. | — |
| `SUCCESS_EMPTY_LESION` | Organ mask was empty (no prostate detected); lesion inference was intentionally skipped and an empty mask written. | Review series selection / organ model. |
| `FAIL_MISSING_MODALITY` | Selector produced 0 matches for at least one of T2 / ADC / HIGHB. See `missing_modalities`. | Usually a genuine data issue (see `series_description_sgh_deid_data.tnt_with_anh_input.csv`). |
| `FAIL_ORGAN_SEG` | Modalities present but `organ.nii.gz` missing. | Organ seg operator failure; check logs. |
| `FAIL_POSTPROCESSING` | Original organ exists but cleaned mask or cleanup metrics are missing. | Check organ cleanup logs. |
| `FAIL_LESION_SEG` | Organ present but `lesion_mask.nii.gz` missing. | Lesion seg operator failure. |
| `FAIL_CLASSIFIER` | Lesion mask present but `lesions.txt` missing. | Classifier operator failure. |
| `FAIL_REPORTING` | `lesions.txt` exists but the DICOM SR is missing. | Check malformed/missing source DICOM and SR logs. |
| `FAIL_RTSTRUCT` | One or more required RTSTRUCT files are missing. | Check RTSTRUCT generation logs. |
| `FAIL_RUNTIME` | Container exited with non-zero code. | Check stderr; may be partial. |
| `FAIL_EMPTY_OUTPUT` | Exit 0 but no output files at all. | Every selector rejected the input. |
| `SKIPPED_RESUME` | Resume mode: existing complete output re-used, not re-run. | — |

### Reading / summarising the CSV

The CSV is pandas/Excel friendly. Quick status breakdown without pandas:

```bash
python scripts/case_status.py summary /hive/output/batch_audit_log.csv
```

```
Audit CSV: /hive/output/batch_audit_log.csv
Total rows: 3538
  FAIL_EMPTY_OUTPUT            3
  FAIL_MISSING_MODALITY       18
  SUCCESS_COMPLETE          3517
```

With pandas:

```python
import pandas as pd
df = pd.read_csv("batch_audit_log.csv")
df["status"].value_counts()
df.query("status.str.startswith('FAIL')")[
    ["patient_id", "status", "missing_modalities", "reason"]
]
```

### Resume semantics

* Fresh run (no `-r`): both `<output>/` and `batch_audit_log.csv` are wiped.
* Resume (`-r`): the CSV is preserved and appended to. Only cases with
  every current required artifact are skipped.
* Incomplete or legacy case directories are cleared and rerun so stale
  artifacts cannot make a failed retry appear successful.
* An older audit CSV header is migrated to the current column schema
  before a new row is appended.

---

## Expected outputs (single patient)

After a successful run, the output directory will contain:

```
output/
├── t2/t2.nii.gz                        # Intermediate NIfTI (T2)
├── adc/adc.nii.gz                      # Intermediate NIfTI (ADC)
├── highb/highb.nii.gz                  # Intermediate NIfTI (High-B)
├── organ/
│   ├── organ.nii.gz                    # Binary whole-gland prostate mask
│   ├── cleaned_organ.nii.gz            # Largest component + filled holes
│   ├── cleanup_metrics.json            # Organ cleanup audit
│   ├── organ_RTSTRUCT.dcm              # Original organ RT Structure Set
│   └── cleaned_organ_RTSTRUCT.dcm      # Cleaned organ RT Structure Set
├── lesion/
│   ├── fold0–4_lesion_prob.nii.gz      # Per-fold probability maps
│   ├── merged_lesion_prob.nii.gz       # Averaged probability map
│   ├── lesion_mask.nii.gz              # Final binary lesion mask
│   └── lesion_RTSTRUCT.dcm             # DICOM RT Structure Set
├── dicom/                              # Copies of selected DICOM series
│   ├── t2/
│   ├── adc/
│   └── highb/
├── lesions.txt                         # PI-RADS + prostate measurements (YAML)
├── prostate_measurements_SR.dcm        # DICOM TID 1500 measurement report
└── combined_organ_lesion_RTSTRUCT.dcm  # Combined organ + lesion RT Structure Set
```

All four RTSTRUCT files use separate DICOM `CLOSEDPLANAR_XOR` contours.
Pinhole/keyhole encoding is disabled because its artificial connector channels
appear as lines through an ROI in viewers such as CARPL. Contour approximation
is also disabled, and tiny point/line components are converted to valid
pixel-footprint polygons so rasterizing the RTSTRUCT reproduces the NIfTI mask
exactly.

The SR is a DICOM Comprehensive SR referencing the selected T2 study. It
contains cleaned-prostate LR/AP/SI dimensions (mm) and volume (mm³ and cm³),
and is marked `COMPLETE`, `UNVERIFIED`, and not final pending clinical review.

For a strict post-run RTSTRUCT guardrail check:

```bash
python scripts/validate_rtstruct_outputs.py <output-root>
```

This writes `rtstruct_validation.csv` and `rtstruct_validation.json` and exits
non-zero for invalid geometry, references, ROI/mask mismatches, or any
RTSTRUCT-to-NIfTI voxel difference.

---

## HIVE training (separate plain-Docker image)

Training uses `docker/Dockerfile.training`; the working inference image above
is unchanged. The image targets one RTX 6000 Ada GPU per container and keeps
data, manifests, caches, experiments, and models on host-mounted storage.

Build it on an internet-connected machine and optionally save it for HIVE:

```bash
python scripts/build_training_image.py \
  --save docker/training_image.tar
```

On HIVE:

```bash
docker load -i docker/training_image.tar
```

The following example assumes the repository is the current directory and the
full inference results live in the folder shown in the HIVE screenshot. Paths
with spaces must remain quoted.

```bash
HIVE_RUN="$HOME/Documents/Prostate HIVE Inference 4 Jun 2026 - highb fix"

docker run --rm --gpus device=0 --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HIVE_RUN/Interesting outputs":/hive-output:ro \
  -w /workspace \
  prostate-mri-training:latest \
  python scripts/prepare_training_data.py \
    --audit-csv /workspace/audit_assignments.csv \
    --sgh-output-root /hive-output \
    --prostatex-root /workspace/data/preprocessed \
    --output-dir /workspace/artifacts/training
```

Review `artifacts/training/audit_report.md`, `exclusions.csv`, the question
statistics, scanner/model/date breakdowns, and split counts before training.
The manifest paths are container paths, so use the same `/workspace` and
`/hive-output` mounts for every training command.

Train the single organ model:

```bash
docker run --rm --gpus device=0 --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HIVE_RUN/Interesting outputs":/hive-output:ro \
  -w /workspace \
  prostate-mri-training:latest \
  python -m training.train_organ --config configs/organ.yaml \
    --initial-weights models/organ/model.ts
```

Train one lesion fold per job. Repeat with `FOLD=0` through `FOLD=4`; jobs may
be assigned to different GPUs, but each container is restricted to one GPU.

```bash
FOLD=0
docker run --rm --gpus device=0 --shm-size=16g \
  -v "$PWD":/workspace \
  -v "$HIVE_RUN/Interesting outputs":/hive-output:ro \
  -w /workspace \
  prostate-mri-training:latest \
  python -m training.train_lesion --config configs/lesion.yaml --fold "$FOLD" \
    --initial-weights "models/fold${FOLD}/model_best_fold${FOLD}.pth.tar"
```

Checkpoints contain optimizer, scheduler, AMP-scaler, epoch, best metric, and
resolved config. Resume a stopped job with `--resume <model_last...pth.tar>`;
do not also pass `--initial-weights`.

TensorBoard is written beneath each run directory. Serve it from a short-lived
container:

```bash
docker run --rm -p 6006:6006 \
  -v "$PWD":/workspace:ro \
  prostate-mri-training:latest \
  tensorboard --logdir /workspace/experiments --bind_all --port 6006
```

If HIVE is remote, use an SSH port forward rather than opening the port
publicly: `ssh -L 6006:localhost:6006 <hive-host>`.

Once reviewed, run `scripts/package_models.py` as documented in `README.md`.
It creates the exact `organ/model.ts`, `fold0..4/*.pth.tar`, and unchanged
classifier layout expected by inference, plus SHA-256 checksums.
Run the locked test cohort once with `python -m training.evaluate` after all
model choices are frozen; it produces overall, per-case, manufacturer, scanner
model, and study-year metrics.

---

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"
NVIDIA Container Toolkit is not installed. See the Prerequisites section.

### "CUDA error: no CUDA-capable device is detected"
The host GPU driver is too old or missing. Run `nvidia-smi` on the host
to verify the GPU is visible. The host driver must support CUDA ≥ 12.1.

### Organ mask is empty / "Skipping lesion inference"
The input DICOM series might have too few slices or the wrong modality
was selected. Check the intermediate NIfTI files in `output/t2/`,
`output/adc/`, `output/highb/` to verify correct series selection.

### Image is very large (> 10 GB)
This is expected. The CUDA runtime + PyTorch + dependencies are ~8 GB.
The application code itself is < 1 MB.

### Running without GPU (CPU fallback)
```bash
python load_and_run.py -i <input> -o <output> -m <models> --cpu
```
This is supported but **significantly slower** (30+ minutes per patient
vs ~2 minutes on GPU).

---

## Customising the Dockerfile

In general, **no changes are needed** unless you want to:

### Change the CUDA / PyTorch version

```dockerfile
# Base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# PyTorch install
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Change the base image tag and the PyTorch index URL together.
Available PyTorch CUDA builds: `cu118`, `cu121`, `cu124`, `cu126`.
Note: `torch==2.6.0` does **not** have a `cu121` wheel – it only ships for
`cu118`, `cu124`, and `cu126`. Keep torch and the index URL matched.

### Add Python packages

Add them to `docker/requirements.txt` and rebuild:
```bash
python scripts/build_and_save.py
```

### Environment variables

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1   # no .pyc files
ENV PYTHONUNBUFFERED=1          # immediate log output
```

---

## File reference

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Container definition (CUDA + app code) |
| `docker/requirements.txt` | Python deps (torch excluded, handled separately) |
| `docker/docker-compose.yaml` | Compose config with volume mounts |
| `docker/Dockerfile.training` | Separate CUDA training image |
| `docker/requirements-training.txt` | Training-only dependencies |
| `docker/docker-compose.training.yaml` | One-GPU training Compose example |
| `docker/docker_image.tar` | Saved image (generated by build script) |
| `.dockerignore` | Excludes models/data/output from build context |
| `scripts/build_and_save.py` | Build image + save to tar |
| `scripts/build_training_image.py` | Build/save the separate training image |
| `scripts/load_and_run.py` | Load tar + run pipeline (single patient) |
| `scripts/batch_load_and_run.py` | Batch processing with resume + cooldown + audit CSV |
| `scripts/case_status.py` | Per-case status classifier + audit-CSV writer (shared by both `test_batch.sh` and `batch_load_and_run.py`) |
