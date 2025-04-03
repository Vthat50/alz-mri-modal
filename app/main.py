import modal
from fastapi import FastAPI

stub = modal.Stub("alz-test-minimal")
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello World"}

stub.app = app  # Critical for v0.73.141
