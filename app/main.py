import modal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import base64

# Modal 0.73.141 REQUIRES stub definition first
stub = modal.Stub("alz-mri-prod")

# Image config (must come after stub)
image = modal.Image.from_dockerhub("deepmi/fastsurfer:cu124-v2.3.3").pip_install(
    "openai",
    "python-dotenv",
    "fpdf2",
    "gunicorn"
)

# Persistent storage
volume = modal.Volume.persisted("mri-data")

# FastAPI app (defined inline for v0.73.141 compatibility)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# MRI Processing Function
@stub.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("openai-key")]
)
def process_mri(file_contents: bytes, filename: str):
    """Identical to original but with local imports"""
    from fastsurfer import run_fastsurfer
    subject_id = f"sub-{uuid.uuid4().hex[:8]}"
    input_path = f"/data/{filename}"
    
    with open(input_path, "wb") as f:
        f.write(file_contents)
    
    run_fastsurfer(input_path, subject_id)
    return subject_id

# Report Generation Function
@stub.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-key")]
)
def generate_report(subject_id: str, mmse: int, cdr: float, adas_cog: float):
    """Fully preserved functionality with v0.73.141 syntax"""
    from fastsurfer import parse_stats, predict_stage, generate_summary
    biomarkers = parse_stats(subject_id)
    seg_path = f"/data/{subject_id}/mri/aparc+aseg.png"
    
    with open(seg_path, "rb") as f:
        seg_base64 = base64.b64encode(f.read()).decode()
    
    summary = generate_summary(biomarkers, mmse, cdr, adas_cog)
    return {
        "biomarkers": biomarkers,
        "stage": predict_stage(mmse, cdr, adas_cog),
        "summary": summary,
        "segmentation": seg_base64
    }

# Upload Endpoint (v0.73.141 webhook syntax)
@stub.webhook(method="POST")
async def upload_mri(file: UploadFile = File(...)):
    """Identical to original but uses .call()"""
    subject_id = process_mri.call(await file.read(), file.filename)
    return {"subject_id": subject_id}

# Analysis Endpoint 
@stub.webhook(method="POST")
async def analyze_scores(
    subject_id: str = Form(...),
    mmse: int = Form(...),
    cdr: float = Form(...),
    adas_cog: float = Form(...)
):
    """Preserves all original form handling"""
    return generate_report.call(subject_id, mmse, cdr, adas_cog)

# REQUIRED FOR v0.73.141 TO EXPOSE FASTAPI
app = stub.app
