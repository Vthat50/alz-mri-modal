import modal
from fastapi import FastAPI, UploadFile, File, Form

# 1. Create stub FIRST (critical in v0.73.141)
stub = modal.Stub("alz-mri-working")

# 2. Define image and volume
image = modal.Image.from_dockerhub("deepmi/fastsurfer:cu124-v2.3.3").pip_install(
    "openai",
    "python-dotenv",
    "fpdf2",
    "gunicorn"
)
volume = modal.Volume.persisted("mri-data-working")

# 3. Create FastAPI app INSIDE a function
def create_app():
    app = FastAPI()
    return app

# 4. Processing functions (all imports inside)
@stub.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("openai-key")]
)
def process_mri(file_contents: bytes, filename: str):
    from fastsurfer import run_fastsurfer
    import uuid
    subject_id = f"sub-{uuid.uuid4().hex[:8]}"
    input_path = f"/data/{filename}"
    with open(input_path, "wb") as f:
        f.write(file_contents)
    run_fastsurfer(input_path, subject_id)
    return subject_id

# 5. Webhook endpoints (v0.73.141 style)
@stub.webhook(method="POST")
def upload(file: UploadFile = File(...)):
    return {"subject_id": process_mri.call(file.file.read(), file.filename)}

@stub.webhook(method="POST")
def analyze(
    subject_id: str = Form(...),
    mmse: int = Form(...),
    cdr: float = Form(...),
    adas_cog: float = Form(...)
):
    from fastsurfer import parse_stats, predict_stage, generate_summary
    import base64
    biomarkers = parse_stats(subject_id)
    with open(f"/data/{subject_id}/mri/aparc+aseg.png", "rb") as f:
        seg_base64 = base64.b64encode(f.read()).decode()
    return {
        "biomarkers": biomarkers,
        "stage": predict_stage(mmse, cdr, adas_cog),
        "summary": generate_summary(biomarkers, mmse, cdr, adas_cog),
        "segmentation": seg_base64
    }

# 6. MUST COME LAST - Bind FastAPI app
app = create_app()
stub.app = app
