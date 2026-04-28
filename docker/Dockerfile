# Prostate MRI Lesion Segmentation – Docker image
# Base: NVIDIA CUDA 12.4 + cuDNN runtime (Ubuntu 22.04, Python 3.10)
#
# CUDA 12.4 is required because the app pins torch==2.6.0, which only ships
# wheels for CUDA 12.4 and 12.6 (no cu121 wheel exists for torch 2.6.0).
#
# Models, input data, and output are volume-mounted at runtime – only the
# application code and its Python dependencies are baked into this image.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# System packages (Python 3.10 ships with Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# ---- PyTorch (CUDA 12.4) ----
# Installed first as a separate layer so Docker can cache the ~2 GB download.
# Uses the official PyTorch wheel index for the CUDA-enabled build.
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# ---- Remaining Python dependencies ----
# docker/requirements.txt mirrors prostate_mri_lesion_seg_app/requirements.txt
# but EXCLUDES torch (already installed above with CUDA support).
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Application code ----
# Models, data, and output are NOT included; they are volume-mounted.
COPY prostate_mri_lesion_seg_app/ /app/prostate_mri_lesion_seg_app/

# Default entrypoint runs the MONAI Deploy pipeline.
# Override CMD to change input/output/model paths.
ENTRYPOINT ["python", "/app/prostate_mri_lesion_seg_app"]
CMD ["-i", "/input", "-o", "/output", "-m", "/models"]
