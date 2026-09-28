#!/bin/sh
set -e

mkdir -p "${STORAGE_DIR}"

echo "Starting DICOM Listener..."
echo "AE Title : ${DICOM_AE}"
echo "Port     : ${DICOM_PORT}"
echo "Storage  : ${STORAGE_DIR}"

exec python3 /app/process_received.py
