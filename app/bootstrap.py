import modal
import importlib.util
from pathlib import Path

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

stub = modal.Stub("alz-mri-final-attempt")

# Manually load your main.py
main = load_module("main", Path(__file__).parent / "main.py")

# Explicitly bind the app
stub.app = main.app
