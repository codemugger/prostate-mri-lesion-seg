# Local CARPL result-receiver stub (development/testing only; not shipped).
#
# Build context: carpl/

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CARPL_LIB_DIR=/opt/carpl/lib \
    DICOM_PORT=4242 \
    DICOM_AE=CARPL-RECIEVE \
    STORAGE_DIR=/data/test_carpl

COPY requirements-carpl.txt /tmp/requirements-carpl.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-carpl.txt

RUN mkdir -p /opt/carpl/lib /data/test_carpl
COPY common/carpl_common.py /opt/carpl/lib/
COPY carpl-receiver/receiver.py /opt/carpl/
RUN chmod -R a+rX /opt/carpl

EXPOSE 4242
CMD ["python", "/opt/carpl/receiver.py"]
