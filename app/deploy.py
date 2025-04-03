import modal
import os

# 1. Use correct stub syntax for v0.73.141
stub = modal.Stub("alz-mri-final")

# 2. Create image using old API syntax
image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "uvicorn",
    "python-multipart",
    "openai",
    "python-dotenv",
    "fpdf2",
    "gunicorn"
)

# 3. Volume setup
volume = modal.Volume.persisted("mri-final-vol")

# 4. Deploy function
@stub.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openai-key")]
)
def deploy():
    import subprocess
    subprocess.run([
        "uvicorn", 
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

# 5. Required app binding
stub.app = lambda: None
