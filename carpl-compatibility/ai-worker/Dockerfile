# PyTorch image
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

COPY app ./app
COPY src ./src
COPY config ./config
COPY components ./components
COPY inference.py .
COPY models ./models
COPY ai-worker/worker.py .

RUN ls -R

EXPOSE 5000

CMD ["python", "worker.py"]