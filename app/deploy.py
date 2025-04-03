import modal
import subprocess
from pathlib import Path

stub = modal.Stub("alz-mri-direct-deploy")

# 1. Deploy the raw FastAPI app
@stub.function(
    image=modal.Image.from_dockerhub("deepmi/fastsurfer:cu124-v2.3.3")
    .pip_install("fastapi", "uvicorn", "python-multipart")
)
def deploy():
    # Launch FastAPI server directly
    cmd = [
        "uvicorn", 
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1"
    ]
    subprocess.run(cmd)

# 2. Manually expose endpoints
@stub.webhook(method="POST")
def upload(file: UploadFile = File(...)):
    # Your existing upload logic here
    pass

# 3. Required for v0.73.141
stub.app = lambda: None
