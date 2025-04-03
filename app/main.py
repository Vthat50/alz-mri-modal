import modal

# 1. Use correct stub syntax for v0.73.141
stub = modal.Stub("alz-mri-final-working")

# 2. Create image using PROVEN syntax for v0.73.141
image = modal.Image.from_registry(
    "deepmi/fastsurfer:cu124-v2.3.3",
    add_python="3.10"
).pip_install(
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
    secrets=[modal.Secret.from_name("openai-key")],
    timeout=3600
)
def deploy():
    import subprocess
    subprocess.run([
        "uvicorn", 
        "app.main:app",  # Changed to explicit module path
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1"
    ])

# 5. Required app binding
stub.app = lambda: None
