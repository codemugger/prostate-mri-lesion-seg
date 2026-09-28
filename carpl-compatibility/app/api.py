import torch, os, shutil, threading
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Body
import uuid
from pydantic import BaseModel

from inference import run_inference
from src.config import Parser
from src.model import create_model
from src.checkpoint import load_model
from src.logger import setup_logger

app = FastAPI()
job_store = {}

conf = Parser.from_yaml("config/inference.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger, log_path = setup_logger(run_name=f"{conf.run_name}_API")
logger.info(f"Using device: {device}")
logger.info(f"Starting run {conf.run_name}_{conf.timestamp} through API...")

# initialise model
model = create_model(conf)
model = load_model(model, conf.model_path).to(device)
logger.info(f"Loaded model weights from {conf.model_path}")

class JobRequest(BaseModel):
    job_id: str

def process_job(job_id, model):
    job_store[job_id]["status"] = "Processing"
    
    try:
        logger.info(f"Starting Inference for {job_id}...")
        results = run_inference(conf, model, device, input_dir=job_id)
        job_store[job_id]["results"] = results
        job_store[job_id]["status"] = "Completed"
        logger.info(f"Completed Inference for {job_id}...")
        logger.info(f"Results for {job_id}: {results}")
    
    except Exception as e:
        job_store[job_id]["status"] = "Error"
        job_store[job_id]["error"] = str(e)
        logger.info(f"Encountered Error for {job_id}: {e}...")

def run_inference_async(job_id, model):
    thread = threading.Thread(target=process_job, args=(job_id, model))
    thread.start()

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(conf.image_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    file_path = os.path.join(job_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    job_store[job_id] = {
        "status": "Pending",
        "results": None,
        "dir": job_dir
    }

    run_inference_async(job_id, model)

    return {
        "job_id": job_id,
        "message": "File uploaded successfully and queued for processing."
    }

@app.post("/api/results")
async def get_results(request: JobRequest):
    job_id = request.job_id

    job_dir = f"{conf.image_dir}/{job_id}/"
    if job_id not in job_store:
        return {"error": f"Job {job_id} not found."}

    job = job_store[job_id]

    if job["status"] != "Completed":
        return {
            "status": job["status"],
            "job_id": job_id,
            "message": job.get("error", "Job is still pending")
        }

    findings = []
    df = job["results"]

    for _, row in df.iterrows():
        findings.append({
            "name": row["uid"],
            "probability": float(row["probability"]),
            "prediction": int(row["prediction"])
        })
    shutil.rmtree(job_dir)

    return {
        "findings": findings,
        "status": "Completed",
        "job_id": job_id
    }