"""Presentation-only layout repair; consumes saved errors, changes no metrics/models."""
from pathlib import Path
import json, sys, os
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root)); os.environ.setdefault('MPLCONFIGDIR',str(root/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import read_idx
errors=json.loads((root/'results/errors.json').read_text())['examples']
images=read_idx(root/'data/raw/t10k-images-idx3-ubyte.gz')
fig,axes=plt.subplots(2,4,figsize=(11,8.3))
for ax,e in zip(axes.flat,errors):
 ax.imshow(images[e['test_index']],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
 ax.set_xticks([]);ax.set_yticks([])
 ax.set_title(f"True: {e['true_class']}\nPredicted: {e['predicted_class']}",fontsize=10,pad=9)
 ax.set_xlabel(f"Index {e['test_index']}  /  pair errors: {e['pair_count']}",fontsize=8,labelpad=7)
 for spine in ax.spines.values():spine.set_visible(False)
fig.subplots_adjust(left=.03,right=.99,bottom=.13,top=.86,wspace=.14,hspace=.55)
fig.suptitle('Exploratory errors: CNN + dropout, predeclared seed 11',fontsize=14,fontweight='bold',y=.965)
fig.text(.5,.045,'First test image in each of the eight most frequent directed confusion pairs; indices are zero-based.\nSelection emphasizes common failures and does not represent all image appearances.',ha='center',fontsize=9,color='#52657A')
for ext in ['pdf','png']: fig.savefig(root/f'figures/representative_errors.{ext}',dpi=220,facecolor='white',bbox_inches='tight')
plt.close(fig)
print('Refined error-image spacing from unchanged saved examples.')
