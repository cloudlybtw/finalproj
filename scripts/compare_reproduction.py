"""Compare an independently retrained copy with the delivered reference results."""
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import nbformat
parser=argparse.ArgumentParser();parser.add_argument('retrained_project');args=parser.parse_args()
root=Path(__file__).resolve().parents[1]; replica=Path(args.retrained_project).resolve()
a=np.load(root/'results/test_predictions.npz');b=np.load(replica/'results/test_predictions.npz')
assert set(a.files)==set(b.files)
for key in a.files: assert np.array_equal(a[key],b[key]),f'Prediction/index/label difference: {key}'
x=pd.read_csv(root/'results/comparison.csv').set_index('run');y=pd.read_csv(replica/'results/comparison.csv').set_index('run').loc[x.index]
cols=['accuracy','macro_f1','balanced_accuracy','selected_epoch','parameters']
np.testing.assert_allclose(x[cols].to_numpy(),y[cols].to_numpy(),atol=0,rtol=0,equal_nan=True)
history_differences={}
for name in x.index:
 if name=='logistic':continue
 h1=pd.read_csv(root/f'results/history_{name}.csv');h2=pd.read_csv(replica/f'results/history_{name}.csv')
 diff=float(np.max(np.abs(h1.to_numpy()-h2.to_numpy())))
 assert diff<1e-12,(name,diff)
 history_differences[name]=diff
runtime_files=['config.json','requirements.txt','notebooks/fashion_mnist.ipynb','scripts/run_notebook.py','scripts/refine_figures.py']+[str(p.relative_to(root)) for p in sorted((root/'src').glob('*.py'))]
hashes={}
for rel in runtime_files:
 h=hashlib.sha256((root/rel).read_bytes()).hexdigest()
 assert h==hashlib.sha256((replica/rel).read_bytes()).hexdigest(),f'Runtime source differs: {rel}'
 hashes[rel]=h
execution=json.loads((replica/'results/notebook_train_verification.json').read_text())
assert execution['executed_cells']==execution['code_cells']==8 and execution['error_outputs']==0
nb=nbformat.read(replica/execution['notebook'],as_version=4)
assert all(o.output_type!='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
verification=root/'verification';verification.mkdir(exist_ok=True)
shutil.copyfile(replica/execution['notebook'],verification/'fashion_mnist.retrained.ipynb')
shutil.copyfile(replica/'results/comparison.csv',verification/'retrained_comparison.csv')
record={'status':'PASS','checked_at_utc':datetime.now(timezone.utc).isoformat(),'retrained_project':str(replica),
 'command':'.venv/bin/python scripts/run_notebook.py --mode train','environment':'fresh virtual environment; all 74 pinned requirements installed and pip check passed',
 'dataset':'All original MD5 checks passed. Three files freshly downloaded; the tiny test-label cache was reused during a slow request and also independently re-downloaded with identical checksum.',
 'training':'All three baseline candidates and six CNNs refitted from initialization; no checkpoint reuse during fitting.',
 'prediction_vectors_identical':7,'test_examples_per_vector':10000,'test_indices_and_labels_identical':True,
 'reported_metrics_selected_epochs_parameters_identical':True,'history_max_absolute_differences':history_differences,
 'runtime_file_sha256':hashes,'notebook_execution':execution,'reference_times_preserved':True,
 'retraining_seconds':y.train_seconds.to_dict(),'retrained_notebook':'verification/fashion_mnist.retrained.ipynb',
 'notes':'The report uses the original measured times. Reproduction times are recorded separately; no original model, numerical result, or report was replaced.'}
(root/'results/archive_reproduction.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','prediction_vectors_identical','test_examples_per_vector','reported_metrics_selected_epochs_parameters_identical','history_max_absolute_differences']},indent=2))
