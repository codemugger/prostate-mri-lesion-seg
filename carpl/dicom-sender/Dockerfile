# Prostate MRI CARPL DICOM sender (Storage SCU -> CARPL result receiver)
#
# Build context: carpl/

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CARPL_LIB_DIR=/opt/carpl/lib \
    DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-carpl.txt /tmp/requirements-carpl.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-carpl.txt

RUN mkdir -p /opt/carpl/lib /data
COPY common/carpl_common.py common/healthcheck.py /opt/carpl/lib/
COPY dicom-sender/sender.py tools/carpl_ctl.py /opt/carpl/
RUN chmod -R a+rX /opt/carpl

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "/opt/carpl/lib/healthcheck.py", "--service", "dicom-sender", "--max-age", "60"]

CMD ["python", "/opt/carpl/sender.py"]
