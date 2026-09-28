FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DICOM_PORT=11113 \
    DICOM_AE=CARPLLISTENER \
    STORAGE_DIR=/data/test_carpl

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /data/test_carpl
WORKDIR /app

RUN pip install --no-cache-dir pydicom pynetdicom

COPY receiver.py /app/receiver.py

EXPOSE 11113
VOLUME ["/data/test_carpl"]

CMD ["python", "receiver.py"]
