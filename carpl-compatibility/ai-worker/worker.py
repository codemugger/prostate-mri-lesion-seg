import json
import os
import shutil

import redis
import torch
from datetime import datetime

from inference import run_inference
from src.config import Parser
from src.model import create_model
from src.checkpoint import load_model
from src.logger import setup_logger

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
INPUT_DIR = os.environ.get("INPUT_DIR", "/data/incoming")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/output")

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
    socket_timeout=None,        # no timeout on blocking reads
    socket_connect_timeout=10,  # fail fast if Redis is unreachable at startup
)

conf = Parser.from_yaml("config/inference.yaml")
conf.image_dir = INPUT_DIR + "/"
conf.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger, _ = setup_logger(run_name="ai-worker")
logger.info(f"Using device: {device}")

model = create_model(conf)
model = load_model(model, conf.model_path).to(device)
logger.info("Model loaded. Waiting for jobs...")

while True:
    result = redis_client.blpop("dicom_jobs", timeout=30)
    if result is None:
        continue
    _, raw = result
    job = json.loads(raw)

    job_uid = job["job_uid"]
    input_path = job["input_path"]
    output_path = job["output_path"]

    logger.info(f"[{job_uid}] Processing {input_path}")

    job_image_dir = os.path.join(INPUT_DIR, job_uid)
    os.makedirs(job_image_dir, exist_ok=True)

    try:
        shutil.copy2(input_path, job_image_dir)
        os.makedirs(output_path, exist_ok=True)

        results = run_inference(conf, model, device, input_dir=job_uid)
        results_csv_path = os.path.join(output_path, "results.csv")
        results.to_csv(results_csv_path, index=False)

        # Move the original DICOM from incoming to the job output folder
        output_dicom_path = os.path.join(output_path, os.path.basename(input_path))
        shutil.move(input_path, output_dicom_path)

        logger.info(f"[{job_uid}] Completed. Results saved to {output_path}")

        sender_job = {
            "job_uid": job_uid,
            "dicom_path": output_dicom_path,
            "results_csv": results_csv_path,
            "study_uid": job.get("study_uid", "unknown"),
            "calling_ae": job.get("calling_ae", "unknown"),
        }
        redis_client.rpush("dicom_sender_jobs", json.dumps(sender_job))
        logger.info(f"[{job_uid}] Queued sender job.")

    except Exception as e:
        logger.error(f"[{job_uid}] Failed: {e}")

    finally:
        shutil.rmtree(job_image_dir, ignore_errors=True)
