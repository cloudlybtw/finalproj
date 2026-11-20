"""Build a submission archive without data, environments, or private self-assessment."""
from pathlib import Path
import hashlib, json, zipfile
root=Path(__file__).resolve().parents[1]
paths=[]
for name in ['README.md','requirements.txt','config.json','ROBT407_Final_Project_2026.pdf']:
 paths.append(root/name)
for folder,patterns in {'src':['*.py'],'scripts':['*.py'],'notebooks':['fashion_mnist.ipynb','fashion_mnist.executed.ipynb','fashion_mnist.verified.ipynb'],'results':['*.json','*.csv','*.npz'],'checkpoints':['*.pt','*.joblib'],'verification':['*.ipynb','*.csv','*.json'],'figures':['*.pdf','*.png'],'report':['*.tex','*.cls','*.bib','*.pdf','*.png'],'docs':['*.md'],'output/pdf':['report.pdf','slides.pdf']}.items():
 for pattern in patterns:
  paths.extend((root/folder).rglob(pattern) if folder in ['report','figures'] else (root/folder).glob(pattern))
for name in ['setup_tectonic.py','compile_latex.sh','LATEX_TOOLCHAIN.md','TECTONIC-LICENSE','IEEEtran-LPPL.txt']:
 p=root/'tools'/name
 if p.exists(): paths.append(p)
paths=sorted(set(p for p in paths if p.is_file() and p.name not in ['progress.json'] and 'assessment' not in str(p.relative_to(root)).lower()))
required=['output/pdf/report.pdf','output/pdf/slides.pdf','notebooks/fashion_mnist.executed.ipynb','results/comparison.csv']
for rel in required: assert root/rel in paths, f'Missing {rel}'
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
archive=root/'output/ROBT407_code.zip'; archive.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in paths: z.write(p,'robt407-fashion-mnist/'+str(p.relative_to(root)))
 z.writestr('robt407-fashion-mnist/MANIFEST.sha256.json',json.dumps(manifest,indent=2)+'\n')
print(f'{archive}: {len(paths)+1} entries; {archive.stat().st_size/1e6:.2f} MB')
