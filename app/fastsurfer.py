
import os
import shutil
import gzip
import subprocess
import re
from fpdf import FPDF
from typing import Dict, Optional

def run_fastsurfer(nifti_path: str, subject_id: str) -> None:
    """Enhanced FastSurfer execution with error handling and logging"""
    try:
        # Handle path variations
        os.makedirs("/output", exist_ok=True)
        compressed_path = f"/output/{subject_id}_T1w.nii.gz"
        
        # Smart compression handling
        if not nifti_path.endswith(".gz"):
            print(f"🔵 Compressing {nifti_path}...")
            with open(nifti_path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy(nifti_path, compressed_path)

        print(f"🚀 Starting FastSurfer processing for {subject_id}")
        command = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", "/data:/data",
            "-v", "/output:/output",
            "deepmi/fastsurfer:cu124-v2.3.3",
            "--t1", compressed_path,
            "--sid", subject_id,
            "--sd", "/output",
            "--parallel", "--seg_only"
        ]
        
        # Enhanced subprocess execution
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"✅ FastSurfer completed for {subject_id}\nLogs:\n{result.stdout[:500]}...")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FastSurfer failed: {e}\nOutput:\n{e.output}")
        raise RuntimeError(f"FastSurfer processing failed: {e.stderr}") from e

def parse_stats(subject_id: str) -> Dict[str, Optional[float]]:
    """Robust statistics parser with validation"""
    stats_dir = os.path.join("/output", subject_id, "stats")
    metrics = {}
    thickness = []
    
    # Parse aseg stats with error handling
    aseg_path = os.path.join(stats_dir, "aseg+DKT.stats")
    if os.path.exists(aseg_path):
        try:
            with open(aseg_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            metrics[parts[4]] = float(parts[3])
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"⚠️ Error parsing aseg stats: {e}")

    # Parse cortical thickness
    for hemi in ["lh", "rh"]:
        aparc_path = os.path.join(stats_dir, f"{hemi}.aparc.stats")
        if os.path.exists(aparc_path):
            try:
                with open(aparc_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                thickness.append(float(parts[5]))
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                print(f"⚠️ Error parsing {hemi} aparc stats: {e}")

    # Calculate biomarkers with safe defaults
    lh_vol = metrics.get("Left-Hippocampus", 0)
    rh_vol = metrics.get("Right-Hippocampus", 0)
    lv_vol = metrics.get("Left-Lateral-Ventricle", 0)
    rv_vol = metrics.get("Right-Lateral-Ventricle", 0)

    return {
        "Left Hippocampus": lh_vol,
        "Right Hippocampus": rh_vol,
        "Asymmetry Index": safe_divide(abs(lh_vol - rh_vol), max(lh_vol, rh_vol)),
        "Evans Index": safe_divide((lv_vol + rv_vol), (lh_vol + rh_vol)),
        "Average Cortical Thickness": round(sum(thickness)/len(thickness), 2) if thickness else None
    }

def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division with zero handling"""
    return round(numerator / (denominator + 1e-6), 2) if denominator != 0 else 0.0

def predict_stage(mmse: int, cdr: float, adas: float) -> str:
    """Enhanced stage prediction with validation"""
    if not (0 <= mmse <= 30):
        raise ValueError("MMSE must be between 0-30")
    if not (0 <= cdr <= 3):
        raise ValueError("CDR must be between 0-3")
    if not (0 <= adas <= 85):
        raise ValueError("ADAS-Cog must be between 0-85")
    
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI (Mild Cognitive Impairment)"
    elif cdr == 0 and mmse >= 26:
        return "Normal Cognition"
    return "Uncertain - Requires further evaluation"

def clean_text(text: str) -> str:
    """Enhanced text sanitization"""
    replacements = {
        "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...",
        "é": "e", "ñ": "n", "°": " degrees"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r'[^\x00-\xff]', '', text)

def create_pdf(summary_text: str) -> bytes:
    """Improved PDF generation with error handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        
        cleaned_text = clean_text(summary_text)
        for line in cleaned_text.split("\n"):
            if line.startswith("# "):
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, line[2:], ln=1)
                pdf.set_font("Arial", size=12)
            else:
                pdf.multi_cell(0, 8, line)
        
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        print(f"⚠️ PDF generation failed: {e}")
        raise RuntimeError("Failed to generate PDF report") from e

def generate_summary(biomarkers: dict, mmse: int, cdr: float, adas: float) -> str:
    """Enhanced GPT-4 summary with fallback"""
    from openai import OpenAI
    from openai import APIConnectionError, RateLimitError
    
    client = OpenAI()  # API key from Modal secret
    
    prompt = f"""**Patient Cognitive Assessment Results**
    
|| Normal Range | Patient Value |
|---|---|---|
| MMSE | 27-30 | {mmse} |
| CDR | 0 | {cdr} |
| ADAS-Cog | 0-10 | {adas} |

**MRI Biomarkers:**
- Hippocampal Volume (L/R): {biomarkers['Left Hippocampus']:.2f}/{biomarkers['Right Hippocampus']:.2f} mm³
- Hemispheric Asymmetry: {biomarkers['Asymmetry Index']:.2f} (0-0.2 normal)
- Ventricular Enlargement (Evans): {biomarkers['Evans Index']:.2f} (<0.3 normal)
- Cortical Thickness: {biomarkers['Average Cortical Thickness'] or 'N/A'} mm

Generate a clinical report with:
1. **Key Findings** - Bullet points of abnormal results
2. **Differential Diagnosis** - Possible conditions
3. **Confidence Level** - Certainty of Alzheimer's diagnosis (Low/Medium/High)
4. **Recommended Actions** - Next diagnostic steps"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{
                "role": "system",
                "content": "You are a neurology specialist analyzing dementia cases."
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=0.5,  # More conservative output
            max_tokens=2000
        )
        return response.choices[0].message.content
        
    except (APIConnectionError, RateLimitError) as e:
        print(f"⚠️ OpenAI API error: {e}")
        return generate_fallback_summary(biomarkers, mmse, cdr, adas)

def generate_fallback_summary(biomarkers: dict, mmse: int, cdr: float, adas: float) -> str:
    """Local fallback when OpenAI fails"""
    stage = predict_stage(mmse, cdr, adas)
    return f"""Clinical Report (Local Analysis)
    
**Key Findings:**
- Cognitive Stage: {stage}
- Hippocampal Volume: {biomarkers['Left Hippocampus']:.1f} (L) / {biomarkers['Right Hippocampus']:.1f} (R) mm³
- Ventricular Enlargement: {'Abnormal' if biomarkers['Evans Index'] > 0.3 else 'Normal'}

**Recommendations:**
1. Clinical correlation required
2. Follow-up neuropsychological testing
3. Consider CSF biomarker analysis"""
