"""Compact report-specific view of the first four documented error pairs."""
import gzip, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
images=np.frombuffer(gzip.decompress((ROOT/'data/raw/t10k-images-idx3-ubyte.gz').read_bytes())[16:],dtype=np.uint8).reshape(-1,28,28)
errors=json.loads((ROOT/'results/errors.json').read_text())['examples'][:4]
fig,axes=plt.subplots(2,2,figsize=(3.4,3.2))
for ax,item in zip(axes.flat,errors):
 ax.imshow(images[item['test_index']],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
 ax.set_title(f"True: {item['true_class']}\nPred.: {item['predicted_class']}",fontsize=7,pad=3)
 ax.set_xlabel(f"Index {item['test_index']} | count {item['pair_count']}",fontsize=6)
 ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_visible(False)
fig.tight_layout(pad=.3,w_pad=.8,h_pad=1.2)
fig.savefig(ROOT/'report/generated/errors_grid.pdf',bbox_inches='tight')
fig.savefig(ROOT/'report/generated/errors_grid.png',dpi=220,bbox_inches='tight')
plt.close(fig)
