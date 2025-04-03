import modal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastsurfer import run_fastsurfer, parse_stats, predict_stage, generate_summary, create_pdf
import os
import uuid
import base64

# Modal setup (must be named 'stub')
stub = modal.Stub("alz-mri-prod")

# Image configuration
image = modal.Image.from_dockerhub("deepmi/fastsurfer:cu124-v2.3.3").pip_install(
    "openai",
    "python-dotenv",
    "fpdf2",
    "gunicorn"
)

# Volume for persistent storage
volume = modal.Volume.persisted("mri-data")

# FastAPI app creation
def create_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    return app

app = create_app()

# Core processing functions
@stub.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("openai-key")]
)
def process_mri(file_contents: bytes, filename: str):
    subject_id = f"sub-{uuid.uuid4().hex[:8]}"
    input_path = f"/data/{filename}"
    
    with open(input_path, "wb") as f:
        f.write(file_contents)
    
    run_fastsurfer(input_path, subject_id)
    return subject_id

@stub.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-key")]
)
def generate_report(subject_id: str, mmse: int, cdr: float, adas_cog: float):
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

# ASGI application mount
@stub.asgi()
def wrapper():
    @app.post("/upload-mri")
    async def upload(file: UploadFile = File(...)):
        subject_id = process_mri.remote(await file.read(), file.filename)
        return {"subject_id": subject_id}

    @app.post("/analyze-scores")
    async def analyze(
        subject_id: str = Form(...),
        mmse: int = Form(...),
        cdr: float = Form(...),
        adas_cog: float = Form(...)
    ):
        results = generate_report.remote(subject_id, mmse, cdr, adas_cog)
        return results
    
    return app
