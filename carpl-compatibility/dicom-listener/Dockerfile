FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DICOM_PORT=11112 \
    DICOM_AE=CARPLSTORESCP \
    STORAGE_DIR=/data/incoming

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /data/incoming
WORKDIR /app
RUN pip install --no-cache-dir pydicom pynetdicom redis requests
COPY entrypoint.sh /app/entrypoint.sh
COPY process_received.py /app/process_received.py
RUN sed -i 's/\r//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh /app/process_received.py
EXPOSE 11112
VOLUME ["/data"]
ENTRYPOINT ["/app/entrypoint.sh"]