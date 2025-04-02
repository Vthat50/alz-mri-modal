import modal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastsurfer import run_fastsurfer, parse_stats, predict_stage, generate_summary, create_pdf
import os
import uuid
import base64

# Modal setup
image = modal.Image.from_dockerhub("deepmi/fastsurfer:cu124-v2.3.3").pip_install(
    "openai",
    "python-dotenv",
    "fpdf2",
    "gunicorn"
)

stub = modal.Stub("alz-mri", image=image)
volume = modal.Volume.persisted("fastsurfer-data")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@stub.function(
    gpu="T4",
    volumes={"/output": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("openai-key")]
)
def process_mri(file_contents: bytes, filename: str):
    """Process MRI with GPU acceleration"""
    subject_id = f"sub-{uuid.uuid4().hex[:8]}"
    input_path = f"/tmp/{filename}"
    
    # Save and process file
    with open(input_path, "wb") as f:
        f.write(file_contents)
    
    # Run FastSurfer
    run_fastsurfer(input_path, subject_id)
    return subject_id

@stub.function(volumes={"/output": volume})
def get_results(subject_id: str, mmse: int, cdr: float, adas_cog: float):
    """Retrieve processed results"""
    biomarkers = parse_stats(subject_id)
    seg_path = f"/output/{subject_id}/mri/aparc+aseg.png"
    
    with open(seg_path, "rb") as f:
        seg_base64 = base64.b64encode(f.read()).decode()

    return {
        "biomarkers": biomarkers,
        "stage": predict_stage(mmse, cdr, adas_cog),
        "summary": generate_summary(biomarkers, mmse, cdr, adas_cog),
        "segmentation": seg_base64,
        "pdf": base64.b64encode(create_pdf(summary)).decode()
    }

@stub.asgi()
def fastapi_app():
    @app.post("/upload-mri/")
    async def upload_mri(file: UploadFile = File(...)):
        subject_id = process_mri.remote(await file.read(), file.filename)
        return {"message": "Processing started", "subject_id": subject_id}

    @app.post("/analyze-scores/")
    async def analyze_scores(
        subject_id: str = Form(...),
        mmse: int = Form(...),
        cdr: float = Form(...),
        adas_cog: float = Form(...)
    ):
        results = get_results.remote(subject_id, mmse, cdr, adas_cog)
        return {
            "🧠 Clinical Biomarkers": results["biomarkers"],
            "🧬 Disease Stage": results["stage"],
            "📋 GPT Summary": results["summary"],
            "🧠 Brain Segmentation Preview (base64)": results["segmentation"],
            "🧾 PDF Report": "✅ Generated"
        }
    
    return app
