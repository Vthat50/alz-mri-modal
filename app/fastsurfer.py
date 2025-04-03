import os
import shutil
import gzip
import subprocess
import re
from fpdf import FPDF

def run_fastsurfer(nifti_path: str, subject_id: str):
    """Run FastSurfer with GPU acceleration"""
    compressed_path = f"/output/{subject_id}_T1w.nii.gz"
    
    # Compress if needed
    if not nifti_path.endswith(".gz"):
        with open(nifti_path, "rb") as f_in, gzip.open(compressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(nifti_path, compressed_path)

    print(f"🚀 Running FastSurfer on: {compressed_path}")
    command = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", "/output:/output",
        "deepmi/fastsurfer:cu124-v2.3.3",
        "--t1", f"/output/{subject_id}_T1w.nii.gz",
        "--sid", subject_id,
        "--sd", "/output",
        "--parallel", "--seg_only"
    ]
    subprocess.run(command, check=True)

def parse_stats(subject_id: str):
    """Parse FastSurfer output statistics"""
    stats_dir = os.path.join("/output", subject_id, "stats")
    metrics = {}
    thickness = []
    
    # Parse aseg stats
    aseg_path = os.path.join(stats_dir, "aseg+DKT.stats")
    if os.path.exists(aseg_path):
        with open(aseg_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    metrics[parts[4]] = float(parts[3])
                except (ValueError, IndexError):
                    continue

    # Parse cortical thickness
    aparc_path = os.path.join(stats_dir, "lh.aparc.stats")
    if os.path.exists(aparc_path):
        with open(aparc_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        thickness.append(float(parts[5]))
                    except (ValueError, IndexError):
                        continue

    # Calculate biomarkers
    lh_vol = metrics.get("Left-Hippocampus", 0)
    rh_vol = metrics.get("Right-Hippocampus", 0)
    lv_vol = metrics.get("Left-Lateral-Ventricle", 0)
    rv_vol = metrics.get("Right-Lateral-Ventricle", 0)

    return {
        "Left Hippocampus": lh_vol,
        "Right Hippocampus": rh_vol,
        "Asymmetry Index": round(abs(lh_vol - rh_vol) / max(lh_vol, rh_vol + 1e-6), 2),
        "Evans Index": round((lv_vol + rv_vol) / (lh_vol + rh_vol + 1e-6), 2),
        "Average Cortical Thickness": round(sum(thickness)/len(thickness), 2) if thickness else None
    }

def predict_stage(mmse: int, cdr: float, adas: float) -> str:
    """Classify Alzheimer's stage based on cognitive scores"""
    if cdr >= 1 or mmse < 21 or adas > 35:
        return "Alzheimer's"
    elif 0.5 <= cdr < 1 or 21 <= mmse < 26:
        return "MCI"  # Mild Cognitive Impairment
    elif cdr == 0 and mmse >= 26:
        return "Normal"
    return "Uncertain"

def clean_text(text: str) -> str:
    """Sanitize text for PDF generation"""
    replacements = {
        "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "..."
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r'[^\x00-\xff]', '', text)

def create_pdf(summary_text: str) -> bytes:
    """Generate PDF report from GPT summary"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    for line in clean_text(summary_text).split("\n"):
        pdf.multi_cell(0, 10, line)
    
    return pdf.output(dest='S').encode('latin1')

def generate_summary(biomarkers: dict, mmse: int, cdr: float, adas: float) -> str:
    """Generate clinical summary using GPT-4"""
    from openai import OpenAI
    client = OpenAI()  # API key loaded from Modal secret
    
    prompt = f"""Clinical MRI and cognitive assessment results:

**Biomarkers:**
- Left Hippocampus Volume: {biomarkers['Left Hippocampus']:.2f} mm³
- Right Hippocampus Volume: {biomarkers['Right Hippocampus']:.2f} mm³
- Hemispheric Asymmetry Index: {biomarkers['Asymmetry Index']:.2f}
- Evans Ratio (ventricular enlargement): {biomarkers['Evans Index']:.2f}
- Mean Cortical Thickness: {biomarkers['Average Cortical Thickness']:.2f} mm

**Cognitive Scores:**
- MMSE: {mmse}/30
- CDR Global: {cdr}
- ADAS-Cog: {adas}/85

Provide a clinical interpretation in Markdown format with these sections:
1. **Key Findings**
2. **Stage Classification**
3. **Clinical Implications**
4. **Recommended Next Steps**"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message.content
