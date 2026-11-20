"""Execute from a fresh kernel, saving visible outputs even on cell failure."""
import argparse, os, sys, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['train','verify'],default='train'); p.add_argument('--output',default='notebooks/fashion_mnist.executed.ipynb'); args=p.parse_args()
root=Path(__file__).resolve().parents[1]
os.environ['FASHION_MODE']=args.mode
os.environ['MPLCONFIGDIR']=str(root/'tmp/matplotlib'); os.environ['IPYTHONDIR']=str(root/'tmp/ipython')
nb=nbformat.read(root/'notebooks/fashion_mnist.ipynb',as_version=4)
# Explicit interpreter makes this command independent of installed global kernelspecs.
km=KernelManager(kernel_name='python3'); km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,timeout=7200,km=km,resources={'metadata':{'path':str(root)}},record_timing=True)
try: client.execute()
finally:
 out=root/args.output; out.parent.mkdir(parents=True,exist_ok=True); nbformat.write(nb,out)
 if km.has_kernel: km.shutdown_kernel(now=True)
report={'mode':args.mode,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'notebook':str(out.relative_to(root)),'code_cells':len([c for c in nb.cells if c.cell_type=='code']),'executed_cells':len([c for c in nb.cells if c.cell_type=='code' and c.execution_count is not None]),'error_outputs':sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),'outputs_retained':sum(len(c.outputs) for c in nb.cells if c.cell_type=='code'),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(root/'results'/f'notebook_{args.mode}_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Executed {args.mode} notebook: {out}')
