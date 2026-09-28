FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /data/output /data/test_carpl
WORKDIR /app

RUN pip install --no-cache-dir pydicom pynetdicom redis pandas

COPY sender.py /app/sender.py

CMD ["python", "sender.py"]
