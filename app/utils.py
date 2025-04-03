import os
import re
from typing import Dict, Optional

def validate_nifti_filename(filename: str) -> bool:
    """Check if filename has valid NIfTI extension"""
    return bool(re.match(r'^.*\.nii(\.gz)?$', filename))

def clean_text(text: str) -> str:
    """Sanitize text for PDF/reports"""
    replacements = {"’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r'[^\x00-\xff]', '', text)

def format_biomarkers(biomarkers: Dict) -> Dict:
    """Convert all float values to 2 decimal places"""
    return {k: round(v, 2) if isinstance(v, float) else v 
            for k, v in biomarkers.items()}
