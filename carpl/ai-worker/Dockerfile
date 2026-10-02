# Prostate MRI CARPL ai-worker
#
# Built FROM the already-validated inference image, so the CUDA/PyTorch/MONAI
# environment and application code are the identical layers used for HIVE
# batch inference.  docker/Dockerfile is not fully reproducible (unpinned
# transitive dependencies), so we never rebuild it separately for CARPL.
#
# The only additions are:
#   * redis-py, installed with --target into /opt/carpl/lib so the app's
#     site-packages stay byte-identical (checked by build_carpl_images.py);
#   * the worker loop, the shared helpers and scripts/case_status.py.
#
# Build context: repository root (see Dockerfile.dockerignore allowlist).
#   docker build -f carpl/ai-worker/Dockerfile \
#       --build-arg BASE_IMAGE=prostate-mri-seg:1.0.0 -t prostate-carpl-ai-worker:1.0.0 .

ARG BASE_IMAGE=prostate-mri-seg:1.0.0
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG BASE_IMAGE_ID=unknown
LABEL org.opencontainers.image.title="prostate-carpl-ai-worker" \
      org.opencontainers.image.description="CARPL DICOM-push worker for the unchanged prostate MRI MONAI Deploy pipeline" \
      prostate.carpl.base-image="${BASE_IMAGE}" \
      prostate.carpl.base-image-id="${BASE_IMAGE_ID}"

# HOME=/tmp: services run as an arbitrary PUID without a passwd entry.  The
# pipeline subprocess gets its own per-job HOME/TMPDIR from worker.py.
ENV CARPL_LIB_DIR=/opt/carpl/lib \
    DATA_DIR=/data \
    MODELS_DIR=/models \
    HOME=/tmp

# holoscan ships its shared libraries through wheel-axle: on the first Python
# start it creates the libholoscan_*.so.2 symlinks listed in its symlinks.txt,
# writes axle.done and removes its one-shot .pth hook.  That needs root.  HIVE
# runs the base image as root, so this happens on every run there.  Do it once
# here so the worker can run as a non-root PUID.  build_carpl_images.py checks
# that this is the only change to site-packages.
RUN python -c "import holoscan.core"

RUN python -m pip install --no-cache-dir --no-compile --target /opt/carpl/lib redis==8.1.0

COPY carpl/common/carpl_common.py carpl/common/healthcheck.py scripts/case_status.py /opt/carpl/lib/
COPY carpl/ai-worker/worker.py /opt/carpl/
RUN chmod -R a+rX /opt/carpl

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD ["python", "/opt/carpl/lib/healthcheck.py", "--service", "ai-worker", "--max-age", "120"]

ENTRYPOINT ["python", "/opt/carpl/worker.py"]
CMD []
