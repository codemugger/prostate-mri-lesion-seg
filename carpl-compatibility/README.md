# Osteoporosis Detection — Deployment Guide

Machine learning pipeline for osteoporosis risk classification from chest X-ray DICOM images combined with patient demographics (age and gender). Supports local CLI/API workflows and a fully containerised CARPL deployment pipeline.

---

## Setup

### Step 1: Python virtual environment

Python 3.10 is required.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell
# or
source .venv/Scripts/activate     # Git Bash
```

### Step 2: Install dependencies

```powershell
pip install --no-cache-dir -r requirements.txt -c constraints.txt
pip install --no-cache-dir setuptools==75.1.0 wheel==0.44.0 -c constraints.txt
pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118 -c constraints.txt
```

### Step 3: Data folders

Two separate data roots exist depending on the workflow:

- **`local_data/`** — used by local CLI, API, training, and evaluation workflows
- **`data/`** — used exclusively by the Docker/CARPL pipeline (managed automatically by the containers)

Create the local data folders:

```powershell
mkdir local_data\evaluation\images, local_data\inference\images\local, local_data\training\images
```

### Project structure

```
project_root/
├── ai-worker/              # Docker service — AI inference worker
├── app/                    # FastAPI backend + Streamlit frontend
├── carpl-receiver/         # Docker service — local DICOM stub (testing only)
├── components/             # Model weights and fitted pipelines (must not be moved)
│   ├── resnet_weights.pth
│   ├── isotonic_calibrator.pkl
│   └── tabular_pipeline.pkl
├── config/                 # YAML configuration files
├── data/                   # CARPL Docker pipeline data (auto-managed by containers)
│   ├── incoming/           # DICOMs received from CARPL (transient)
│   ├── output/             # Intermediate job workspace (auto-deleted after send)
│   ├── test_carpl/         # Annotated DICOMs land here in local test mode
│   ├── evaluation/images/
│   ├── inference/images/local/
│   └── training/images/
│
├── dicom-listener/         # Docker service — DICOM C-STORE receiver (Storage SCP)
├── dicom-sender/           # Docker service — attaches findings, sends result back
├── outputs/                # Training/evaluation results
├── src/                    # Core ML source code
├── scripts/                # Batch helper scripts
└── requirements.txt
```

---

## Workflows

### (A) Local CLI Inference

1. Place model weights and pipelines in `components/`:
   - `resnet_weights.pth`
   - `tabular_pipeline.pkl`
   - `isotonic_calibrator.pkl` (optional — applied if present)

2. Place DICOM images in `local_data/inference/images/local/`

3. Run:
```bash
python inference.py
```

Output CSVs are written to `outputs/inference/`.

---

### (B) API + Streamlit UI

1. Start both services via the batch script:
```powershell
scripts\run_streamlit_inference.bat
```

Or manually in two terminals:
```bash
uvicorn app.api:app          # FastAPI on :5000
streamlit run app/frontend.py  # Streamlit on :8501
```

2. Open `http://127.0.0.1:8501` in a browser and upload a DICOM file.

---

### (C) Retraining

Place training data in:
```
local_data/training/images/          # DICOM files
local_data/training/tabular_data_train.csv   # columns: uid, target
local_data/training/tabular_data_val.csv     # columns: uid, target
```

Run:
```bash
python training.py
```

Results and checkpoints are saved under `outputs/train/` in a timestamped subfolder.

---

### (D) Evaluation

Place evaluation data in:
```
local_data/evaluation/images/                    # DICOM files
local_data/evaluation/tabular_data_evaluation.csv  # columns: uid, target
```

Run:
```bash
python evaluation.py
```

Results (metrics + 95% bootstrap confidence intervals) are saved under `outputs/evaluation/`.

---

### (E) Model Calibration

Fits an isotonic regression calibrator on top of raw model outputs to improve probability estimates.

Requires a `evaluation_results.csv` from a prior evaluation run. Update `results_csv` in `config/calibration.yaml` to point to it, then run:

```bash
python calibrate.py
```

The fitted calibrator is saved to `components/isotonic_calibrator.pkl`. Metrics (ECE, Brier score, AUC before/after) are written to `outputs/calibration/`.

---

### (F) CARPL Deployment — Containerised DICOM Pipeline

End-to-end containerised pipeline for CARPL deployment. CARPL pushes a DICOM study to the AI system over the network; the AI runs inference, embeds the result as a private DICOM tag, and sends the annotated file back to CARPL — no manual file handling required.

#### How the DICOM flow works

There are two separate DICOM connections:

**1. Inbound — CARPL pushes the study to your AI system**

CARPL connects to your machine and sends the DICOM file via C-STORE. You register your machine's details in CARPL's routing page so CARPL knows where to send studies.

| Parameter | Your value | Where it's set |
|---|---|---|
| AE Title | `AI_ENGINE` | `docker-compose.yml` → `dicom-listener` → `DICOM_AE` |
| IP Address | The LAN IP of your server | You provide this to CARPL — find it with `ipconfig` |
| Port | `11112` | `docker-compose.yml` → `dicom-listener` → `ports` |

**2. Outbound — your AI sends the result back to CARPL**

After inference, your `dicom-sender` connects to CARPL's own receiver and C-STOREs the annotated DICOM back. CARPL gives you their details; they are already configured in `docker-compose.yml`.

| Parameter | Value | Where it's set |
|---|---|---|
| AE Title | `CARPL_DICOM` | `docker-compose.yml` → `dicom-sender` → `DEST_AE` |
| IP Address | `<carpl-ip-address>` | `docker-compose.yml` → `dicom-sender` → `DEST_HOST` |
| Port | `11112` | `docker-compose.yml` → `dicom-sender` → `DEST_PORT` |

```
CARPL platform                         Your server
      │                                      │
      │  pushes study via C-STORE            │
      │  to AI_ENGINE @ <your-ip>:11112      │
      └─────────────────────────────────────►│
                                        dicom-listener
                                             │  queues job (Redis)
                                             ▼
                                         ai-worker
                                         (inference)
                                             │  queues result (Redis)
                                             ▼
                                        dicom-sender
                                             │
      ◄──────────────────────────────────────┘
      │  sends annotated DICOM via C-STORE
      │  to CARPL_DICOM @ <carpl-ip-address>:11112
CARPL platform
```

#### Pipeline services

| Service | Role | Port |
|---|---|---|
| `dicom-listener` | Receives incoming DICOMs from CARPL via C-STORE SCP; queues jobs to Redis | 11112 |
| `redis` | Job queue between all services | 6379 |
| `ai-worker` | Pulls jobs from Redis, runs model inference, queues sender jobs | — |
| `dicom-sender` | Pulls sender jobs, embeds findings as a private DICOM tag, sends result | — |
| `carpl-receiver` | Local stub that mimics CARPL's receiver — only used for local testing | 11113 |

#### Local test mode vs production mode

The `dicom-sender` has two modes, controlled by `SENDER_MODE` in `docker-compose.yml`:

| Mode | What happens | When to use |
|---|---|---|
| `local` | C-STOREs annotated DICOM to the local `carpl-receiver` stub — result lands in `data/test_carpl/` | Local testing and development |
| `production` | C-STOREs annotated DICOM to CARPL's real receiver at `<carpl-ip-address>:11112` | Live CARPL deployment |

To switch modes, change one line in `docker-compose.yml` under the `dicom-sender` environment block:

```yaml
SENDER_MODE: local        # saves result to data/local_output/ (default)
SENDER_MODE: production   # C-STOREs result to CARPL's server
```

#### Findings tag format

The AI result is embedded in the returned DICOM as a private tag under creator `CARPL_AI`:

| Tag | Content |
|---|---|
| `(0013,0010)` | `CARPL_AI` — private block reservation |
| `(0013,1001)` | JSON findings payload (VR: `LT`) |

Payload format:
```json
{
  "findings": [
    {"name": "osteoporosis", "probability": 0.8958}
  ]
}
```

`probability` is the calibrated model output rounded to 4 decimal places.

To read the tag from any Python environment:
```python
import json, pydicom

ds = pydicom.dcmread("path/to/annotated.dcm")
block = ds.private_block(0x0013, "CARPL_AI")
findings = json.loads(block[0x01].value)

print(findings["findings"][0]["name"])         # "osteoporosis"
print(findings["findings"][0]["probability"])  # e.g. 0.8958
```

#### Prerequisites

- Docker and Docker Compose installed on the server
- Model weights and fitted pipelines present in `components/`:
  - `components/resnet_weights.pth`
  - `components/tabular_pipeline.pkl`
  - `components/isotonic_calibrator.pkl` (optional)

#### Running the pipeline

1. Create the required data directories:
```powershell
mkdir data\incoming, data\output, data\test_carpl
```

2. Build and start all services:
```powershell
docker compose up --build -d
```

3. Check all services are running:
```powershell
docker compose ps
```

4. Send a test DICOM to the listener (from any C-STORE SCU):
```bash
dcmsend localhost 11112 -aet MY_SCU -aec AI_ENGINE path/to/image.dcm
```

In local mode, the annotated DICOM appears in `data/local_output/` once all stages complete.

Monitor logs at any point:
```powershell
docker compose logs -f               # all services
docker compose logs -f dicom-sender  # one service only
```

#### Switching to production

1. Set `SENDER_MODE: production` in `docker-compose.yml` under `dicom-sender`
2. Register your machine's details on the CARPL routing page (AE Title: `AI_ENGINE`, your server's IP, port `11112`)
3. Restart the sender service:
```powershell
docker compose up -d --no-deps dicom-sender
```
