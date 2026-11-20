"""Independent saved-output checks; does not fit or select any model."""
from pathlib import Path
import hashlib, json, argparse
import numpy as np
import pandas as pd
import nbformat
parser=argparse.ArgumentParser(); parser.add_argument('--skip-manifest',action='store_true',help='Use after regenerating outputs; frozen experimental provenance is still checked.'); args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
manifest=root/'MANIFEST.sha256.json'
if manifest.exists() and not args.skip_manifest:
 for rel,h in json.loads(manifest.read_text()).items():
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h, rel
p=np.load(root/'results/test_predictions.npz'); rows=pd.read_csv(root/'results/comparison.csv')
assert len(rows)==7 and len(p['y_true'])==10000
for row in rows.to_dict('records'):
 y,q=p['y_true'],p[row['run']]
 cm=np.zeros((10,10),dtype=np.int64); np.add.at(cm,(y,q),1)
 assert cm.sum()==10000 and np.all(cm.sum(axis=1)==1000)
 f1=np.mean(2*np.diag(cm)/(cm.sum(axis=0)+cm.sum(axis=1)))
 accuracy=np.trace(cm)/cm.sum()
 assert abs(f1-row['macro_f1'])<1e-12
 assert abs(accuracy-row['accuracy'])<1e-12
 assert abs(np.mean(np.diag(cm)/cm.sum(axis=1))-row['balanced_accuracy'])<1e-12
for variant in ['cnn','dropout']:
 r=rows[rows.variant==variant]; s=pd.read_csv(root/'results/summary.csv').set_index('variant').loc[variant]
 assert len(r)==3 and set(r.seed)=={11,22,33}
 assert np.isclose(r.macro_f1.mean(),s.macro_f1_mean,rtol=0,atol=1e-12)
 assert np.isclose(r.macro_f1.std(ddof=1),s.macro_f1_sd,rtol=0,atol=1e-12)
nb=nbformat.read(root/'notebooks/fashion_mnist.executed.ipynb',as_version=4)
cells=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert all(o.output_type!='error' for c in cells for o in c.outputs)
assert sum(len(c.outputs) for c in cells)>0
frozen=json.loads((root/'results/frozen_protocol.json').read_text())
for rel,h in frozen['sha256'].items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h, rel
print(json.dumps({'status':'PASS','manifest_checked':manifest.exists() and not args.skip_manifest,'runs':7,'examples_each':10000,'metrics_recomputed_independently_from_counts':True,'visible_executed_code_cells':len(cells),'frozen_provenance_intact':True},indent=2))
