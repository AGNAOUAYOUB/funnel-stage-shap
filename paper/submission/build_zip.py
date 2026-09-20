import os
import shutil
import zipfile
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(here)
figures_dir = os.path.join(parent, "figures")
out_zip = os.path.join(parent, "funnel_shap_overleaf.zip")

with tempfile.TemporaryDirectory() as staging:
    figures_staging = os.path.join(staging, "figures")
    os.makedirs(figures_staging, exist_ok=True)
    
    # Copy manuscript files
    shutil.copy(os.path.join(here, "main.tex"), os.path.join(staging, "main.tex"))
    shutil.copy(os.path.join(here, "references.bib"), os.path.join(staging, "references.bib"))
    if os.path.exists(os.path.join(here, "titlepage.tex")):
        shutil.copy(os.path.join(here, "titlepage.tex"), os.path.join(staging, "titlepage.tex"))
    if os.path.exists(os.path.join(here, "highlights.txt")):
        shutil.copy(os.path.join(here, "highlights.txt"), os.path.join(staging, "highlights.txt"))
    if os.path.exists(os.path.join(here, "README.md")):
        shutil.copy(os.path.join(here, "README.md"), os.path.join(staging, "README.md"))
        
    # Copy figures
    if os.path.exists(figures_dir):
        for f in os.listdir(figures_dir):
            if f.endswith(".pdf"):
                shutil.copy(os.path.join(figures_dir, f), os.path.join(figures_staging, f))
                
    # Create zip
    if os.path.exists(out_zip):
        os.remove(out_zip)
        
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(staging):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, staging)
                zipf.write(abs_file, rel_file)

print(f"Created submission package successfully: {out_zip}")
