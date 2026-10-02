#!/bin/sh
set -e

mkdir -p "${DATA_DIR:-/data}"

echo "Starting prostate MRI CARPL DICOM listener..."
echo "AE Title      : ${DICOM_AE}"
echo "Port          : ${DICOM_PORT}"
echo "Data dir      : ${DATA_DIR:-/data}"
echo "Quiet period  : ${STUDY_QUIET_SECONDS:-60}s"
echo "Modalities    : ${ACCEPT_MODALITIES:-MR}"

exec python3 /opt/carpl/listener.py
