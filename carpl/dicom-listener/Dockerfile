# Prostate MRI CARPL DICOM listener (Storage SCP)
#
# Same skeleton as the CARPL template listener and Hazel's CXR-BMD listener
# (python:3.12-slim, /data volume, entrypoint.sh).  pynetdicom replaces DCMTK
# storescp so study-level triggering, deduplication and the series-per-folder
# layout required by the inference app live in one process.
#
# Build context: carpl/

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CARPL_LIB_DIR=/opt/carpl/lib \
    DATA_DIR=/data \
    DICOM_PORT=11112 \
    DICOM_AE=PROSTATE_AI

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-carpl.txt /tmp/requirements-carpl.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-carpl.txt

RUN mkdir -p /opt/carpl/lib /data
COPY common/carpl_common.py common/healthcheck.py /opt/carpl/lib/
COPY dicom-listener/entrypoint.sh dicom-listener/listener.py /opt/carpl/
RUN sed -i 's/\r$//' /opt/carpl/entrypoint.sh \
    && chmod 0755 /opt/carpl/entrypoint.sh /opt/carpl/listener.py \
    && chmod -R a+rX /opt/carpl

EXPOSE 11112
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD ["python", "/opt/carpl/lib/healthcheck.py", "--service", "dicom-listener", "--max-age", "60", "--echo"]

ENTRYPOINT ["/opt/carpl/entrypoint.sh"]
