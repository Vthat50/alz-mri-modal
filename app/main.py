import modal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastsurfer import run_fastsurfer, parse_stats, predict_stage, generate_summary, create_pdf
import os
import uuid
import base64

# Modal configuration
image = modal.Image.from_dockerhub("deepmi/fastsurfer:cu124-v2.3.3").pip_install(
    "openai==1.12.0",
    "python-dotenv==1.0.0",
    "fpdf2==2.7.4",
    "gunicorn==21.2.0"
)

stub = modal.Stub("alz-mri-prod", image=image)
volume = modal.Volume.persisted("mri-data")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@stub.function(
    gpu="T4",
    volumes={"/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("openai-key")]
)
def process_mri(file_contents: bytes, filename: str):
    """Process MRI scan with FastSurfer"""
    subject_id = f"sub-{uuid.uuid4().hex[:8]}"
    input_path = f"/data/{filename}"
    
    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(file_contents)
    
    # Run FastSurfer processing
    run_fastsurfer(input_path, subject_id)
    return subject_id

@stub.function(
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-key")]
)
def generate_report(subject_id: str, mmse: int, cdr: float, adas_cog: float):
    """Generate full analysis report"""
    # Parse biomarkers
    biomarkers = parse_stats(subject_id)
    
    # Get segmentation preview
    seg_path = f"/data/{subject_id}/mri/aparc+aseg.png"
    with open(seg_path, "rb") as f:
        seg_base64 = base64.b64encode(f.read()).decode()
    
    # Generate components
    return {
        "biomarkers": {
            "Left Hippocampus": biomarkers["Left Hippocampus"],
            "Right Hippocampus": biomarkers["Right Hippocampus"],
            "Asymmetry Index": biomarkers["Asymmetry Index"],
            "Evans Index": biomarkers["Evans Index"],
            "Average Cortical Thickness": biomarkers["Average Cortical Thickness"]
        },
        "stage": predict_stage(mmse, cdr, adas_cog),
        "summary": generate_summary(biomarkers, mmse, cdr, adas_cog),
        "segmentation": seg_base64,
        "pdf_report": base64.b64encode(create_pdf(summary)).decode()
    }

@stub.asgi(live=True)  # Enable live reloading during development
def web_app():
    @app.post("/api/upload")
    async def upload_mri(file: UploadFile = File(...)):
        try:
            subject_id = process_mri.remote(
                await file.read(),
                file.filename
            )
            return {
                "status": "success",
                "subject_id": subject_id,
                "message": "MRI processing started"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    @app.post("/api/analyze")
    async def analyze_results(
        subject_id: str = Form(...),
        mmse: int = Form(...),
        cdr: float = Form(...),
        adas_cog: float = Form(...)
    ):
        try:
            report = generate_report.remote(
                subject_id,
                mmse,
                cdr,
                adas_cog
            )
            return {
                "status": "success",
                "data": report
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app
