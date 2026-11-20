"""Create the local submission slides PDF from verified numerical results.

No model fitting, selection, source changes, or numerical-result edits occur here.
Team members may replace TEAM_LINE and the contribution slide before submission.
"""
from pathlib import Path
import os, sys, json, io
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader
from src.data import read_idx, LABELS

TEAM_LINE='ROBT407 project team | Names to be completed by the team'
W,H=960,540
INK='#142F40'; MUTED='#526777'; TEAL='#147D92'; ORANGE='#CB6C32'; RULE='#D8E1E7'
ASSETS=ROOT/'tmp/slides_assets'; ASSETS.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'output/pdf/slides.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
comparison=pd.read_csv(ROOT/'results/comparison.csv')
summary=pd.read_csv(ROOT/'results/summary.csv').set_index('variant')
pairs=pd.read_csv(ROOT/'results/matched_seed_deltas.csv')
classes=pd.read_csv(ROOT/'results/class_metrics.csv')
audit=json.loads((ROOT/'results/data_audit.json').read_text())
tuning=json.loads((ROOT/'results/baseline_tuning.json').read_text())
checks=json.loads((ROOT/'results/checks.json').read_text())
env=json.loads((ROOT/'results/environment.json').read_text())
config=json.loads((ROOT/'config.json').read_text())
errors=json.loads((ROOT/'results/errors.json').read_text())
mean_delta=float(pairs.delta_macro_f1_percentage_points.mean())

# Data charts use saved numerical evidence. They are not decorative illustrations.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':17,'axes.labelsize':14,
 'xtick.labelsize':13,'ytick.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':RULE,'text.color':INK,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'grid.color':'#E4EAF0','pdf.fonttype':42})
colors=[TEAL,ORANGE,'#775DA6']
fig,axes=plt.subplots(1,2,figsize=(12.1,4.0),sharey=True)
for ax,variant,title in zip(axes,['cnn','dropout'],['CNN','CNN + dropout']):
 for seed,col in zip(config['seeds'],colors):
  h=pd.read_csv(ROOT/f'results/history_{variant}_{seed}.csv')
  ax.plot(h.epoch,h.train_loss_online,'--',color=col,linewidth=2)
  ax.plot(h.epoch,h.val_loss,color=col,linewidth=2)
 ax.set_title(title); ax.set_xlabel('Epoch');ax.grid(alpha=.7);ax.xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].set_ylabel('Cross-entropy')
handles=[Line2D([0],[0],color=c,label=f'Seed {s}',linewidth=2) for s,c in zip(config['seeds'],colors)]
handles.extend([Line2D([0],[0],color=MUTED,linestyle='--',label='Training (online)'),Line2D([0],[0],color=MUTED,label='Validation (eval)')])
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=12)
fig.tight_layout(rect=(0,.13,1,1));fig.savefig(ASSETS/'curves.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(12.0,4.0))
for offset,variant,col,label in [(-.17,'cnn',TEAL,'CNN'),(.17,'dropout',ORANGE,'CNN + dropout')]:
 g=classes[classes.variant==variant].groupby('class_id').f1
 ax.errorbar(np.arange(10)+offset,g.mean(),yerr=g.std(ddof=1),fmt='o',color=col,markersize=7,capsize=4,label=label,linewidth=1.8)
ax.set_xticks(np.arange(10),['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'],rotation=25,ha='right')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(.65,1.01);ax.set_ylabel('Mean class F1');ax.grid(axis='y');ax.legend(loc='lower right',frameon=False)
fig.tight_layout();fig.savefig(ASSETS/'classes.png',dpi=180);plt.close(fig)

cm=pd.read_csv(ROOT/'results/confusion_dropout_11.csv',index_col=0).to_numpy()
fig,ax=plt.subplots(figsize=(5,5))
ax.imshow(cm/1000,cmap='Blues',vmin=0,vmax=1)
for i in range(10):
 for j in range(10):
  if cm[i,j]: ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11.5,color='white' if cm[i,j]>550 else INK)
ax.set_xticks(range(10));ax.set_yticks(range(10));ax.set_xlabel('Predicted class ID');ax.set_ylabel('True class ID');ax.tick_params(length=0)
fig.tight_layout();fig.savefig(ASSETS/'confusion.png',dpi=180);plt.close(fig)

c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('ROBT407: Classification-head dropout on Fashion-MNIST');c.setAuthor('ROBT407 project team')
titles=[]
def text(s,x,y,w,size=22,color=INK,bold=False,leading=None):
 """Place paragraph by its top edge; fail rather than silently spill beyond page."""
 style=ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=leading or size*1.22,textColor=HexColor(color),spaceAfter=0)
 p=Paragraph(s,style);ww,hh=p.wrap(w,H)
 assert y+hh<H-12,(titles[-1] if titles else 'cover',s,y,hh)
 p.drawOn(c,x,H-y-hh)
 return hh

def begin(title):
 titles.append(title);c.setFillColor(white);c.rect(0,0,W,H,fill=1,stroke=0)
 text(title,46,32,870,size=31,bold=True)
 c.setStrokeColor(HexColor(RULE));c.setLineWidth(1);c.line(46,H-88,914,H-88)
 c.setFont('Helvetica',11);c.setFillColor(HexColor(MUTED));c.drawRightString(914,19,str(len(titles)))

def foot(s): text(s,46,497,850,size=11,color=MUTED,leading=13)
def end(): c.showPage()
def lines(items,x=48,y=119,w=850,size=23,gap=23):
 for s in items:
  height=text(s,x,y,w,size=size);y+=height+gap
 return y

def table(data,widths,x=48,y=120,fontsize=19,rowheight=39):
 pstyle=ParagraphStyle('cell',fontName='Helvetica',fontSize=fontsize,leading=fontsize*1.17,textColor=HexColor(INK))
 rows=[[Paragraph(str(v),pstyle) for v in row] for row in data]
 t=Table(rows,colWidths=widths,rowHeights=[rowheight]*len(rows))
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),1.5,HexColor(TEAL)),('LINEBELOW',(0,1),(-1,-1),.5,HexColor(RULE)),('BACKGROUND',(0,0),(-1,0),HexColor('#EDF4F6'))]))
 tw,th=t.wrap(900,H);assert y+th<490,(titles[-1],y,th)
 t.drawOn(c,x,H-y-th)
 return th

def picture(path,x,y,w,h):
 image=Image.open(path); iw,ih=image.size;scale=min(w/iw,h/ih);dw,dh=iw*scale,ih*scale
 c.drawImage(ImageReader(image),x+(w-dw)/2,H-y-dh,width=dw,height=dh,mask='auto')

def pct(v,digits=3): return f'{100*float(v):.{digits}f}%'
def pm(variant,key):
 r=summary.loc[variant]
 return f'{100*r[key+"_mean"]:.3f} ± {100*r[key+"_sd"]:.3f}%'

# 1
begin('Classification-head dropout on Fashion-MNIST')
text('Does dropout improve a small CNN?',48,148,820,size=39,bold=True)
text('A controlled comparison on ten clothing categories',48,218,820,size=25,color=TEAL)
text('ROBT407 · Applied Deep Learning · Fall 2026',48,315,820,size=20,color=MUTED)
text(TEAM_LINE.replace('|','<br/>'),48,369,840,size=18,color=MUTED)
foot('15-minute team presentation. Dataset: Xiao, Rasul and Vollgraf (2017), Fashion-MNIST.')
end()
# 2
begin('Question and hypothesis')
lines(['<b>Question:</b> Does adding dropout to the classification head improve held-out macro-F1? Which garments remain hardest? Macro-F1 averages the ten class F1 scores.',
 '<b>Prediction recorded before training:</b> dropout at p = 0.30 will improve mean macro-F1 by reducing reliance on particular hidden activations.',
 '<b>Possible countereffect:</b> regularization can slow fitting within a fixed 12-epoch budget. A negative result remains informative.'],size=24,gap=27)
foot('Notebook §1 records the prediction and comparison rules before training. Dropout rationale: Srivastava et al. (2014).')
end()
# 3
begin('Dataset and label mapping')
text('Original 2017 IDX release: 70,000 grayscale images, each 28 × 28 pixels.',48,111,865,size=23)
left=[];right=[]
for i,label in enumerate(LABELS): (left if i<5 else right).append(f'<b>{i}</b>  {label}')
lines(left,x=60,y=173,w=400,size=23,gap=12);lines(right,x=503,y=173,w=400,size=23,gap=12)
text('All ten classes included. Original files validated using author-provided MD5 checksums.',48,418,865,size=20,color=TEAL)
foot('Source: Xiao, Rasul and Vollgraf (2017), arXiv:1708.07747v2; github.com/zalandoresearch/fashion-mnist.')
end()
# 4
begin('Fixed partitions and training-only preprocessing')
table([['Partition','Images','Per class','Use'],['Training','48,000','4,800','Fit parameters'],['Validation','12,000','1,200','Select models'],['Official test','10,000','1,000','Final evaluation']],[210,155,160,335],y=113,rowheight=42)
text('Split seed 407. Exact-pixel audit: zero duplicates, zero cross-partition overlap and zero exclusions. No missing pixels or labels.',48,307,865,size=21)
text(f'Normalize using training pixels only: (pixel / 255 - {audit["preprocessing"]["mean"]:.6f}) / {audit["preprocessing"]["std"]:.6f}.',48,377,865,size=21)
text('Rule: preserve test; remove overlapping/conflicting development hashes; keep one per remaining hash. No object IDs; near duplicates may remain.',48,438,865,size=18,color=MUTED)
foot('Evidence: results/data_audit.json and split_indices.npz. Every model uses the same held-out examples.')
end()
# 5
begin('Classical baseline: logistic regression')
text('784 normalized pixels, flattened into one feature vector. Multinomial classification with L2 regularization.',48,115,865,size=24)
table([['C','Validation macro-F1','Iterations'],*[[str(r['C']),pct(r['macro_f1']),str(r['iterations'])] for r in tuning['candidates']]],[180,360,320],y=204,rowheight=42)
text(f'Validation selects C = {tuning["selected"]["C"]}. The baseline has {tuning["parameters"]:,} coefficients and intercepts.',48,390,865,size=22,color=TEAL)
text('All final fits converge. A pre-test solver repair raised the common cap from 500 to 5,000 iterations.',48,438,865,size=18,color=MUTED)
foot('Evidence: baseline_tuning.json and development_log.json. Library: scikit-learn LogisticRegression, L-BFGS solver.')
end()
# 6
begin('Small CNN: architecture and tensor dimensions')
table([['Operation','Output shape','Parameters'],['Input','N × 1 × 28 × 28','0'],['3 × 3 Conv, ReLU, Pool','N × 16 × 14 × 14','160'],['3 × 3 Conv, ReLU, Pool','N × 32 × 7 × 7','4,640'],['Flatten to 1,568, Linear, ReLU','N × 64','100,416'],['Dropout, Linear','N × 10 logits','650']],[390,300,170],y=111,rowheight=43,fontsize=20)
text(f'{checks["parameters"]:,} trainable parameters in both variants. Random initialization; no pretrained weights.',48,390,865,size=24,color=TEAL)
text('Cross-entropy consumes raw logits and integer labels. Adam updates all trainable weights.',48,462,865,size=18)
foot('Evidence: src/study.py, SmallCNN; results/checks.json. Pooling is 2 × 2; convolutions use padding 1.')
end()
# 7
begin('One controlled change and implementation checks')
lines(['<b>Only change:</b> head dropout p = 0 versus 0.30, after the 64-unit ReLU and before the final classifier.',
 '<b>Fixed:</b> split, normalization, architecture, Adam at 0.001, no weight decay or augmentation, batch size 128, 12 epochs.',
 '<b>Paired seeds:</b> 11, 22, 33. Identical initial weights and minibatch order. Highest validation macro-F1 selects each checkpoint; the earliest epoch breaks ties.',
 '<b>Checks pass:</b> shapes, labels, finite loss/gradients, eval-mode dropout, and 100% accuracy on a 64-image training sanity subset.'],size=21,gap=22)
foot('Evidence: config.json and checks.json. The sanity check uses 180 optimizer steps and is discarded before full training.')
end()
# 8
begin('Dropout did not improve mean test macro-F1')
table([['Model','Accuracy','Macro-F1'],['Logistic regression',pct(summary.loc['logistic','accuracy_mean']),pct(summary.loc['logistic','macro_f1_mean'])],['CNN',pm('cnn','accuracy'),pm('cnn','macro_f1')],['CNN + dropout',pm('dropout','accuracy'),pm('dropout','macro_f1')]],[300,280,280],y=124,rowheight=55,fontsize=23)
text(f'Mean paired change: <b>{mean_delta:+.3f} percentage points</b> in macro-F1.',48,377,865,size=28,color=ORANGE)
text('The predicted improvement was not supported under this fixed procedure.',48,429,865,size=23)
foot('CNN values: mean ± sample SD across three seeds. Test n = 10,000. Balanced accuracy equals accuracy on this balanced test set.')
end()
# 9
begin('Every neural seed and checkpoint')
data=[['Model','Seed','Accuracy','Macro-F1','Epoch','Fit time']]
for r in comparison[comparison.variant!='logistic'].to_dict('records'):
 data.append(['CNN' if r['variant']=='cnn' else '+ dropout',int(r['seed']),pct(r['accuracy'],2),pct(r['macro_f1']),int(r['selected_epoch']),f'{r["train_seconds"]:.2f} s'])
table(data,[185,85,145,160,100,185],y=111,fontsize=19,rowheight=40)
delta_text=', '.join(f'{v:+.3f}' for v in pairs.delta_macro_f1_percentage_points)
text(f'Paired macro-F1 changes for seeds 11, 22, 33: <b>{delta_text} pp</b>.',48,414,865,size=22,color=ORANGE)
foot('Evidence: comparison.csv and matched_seed_deltas.csv. Three seeds describe training variability, not population significance.')
end()
# 10
begin('Learning curves and compute budget')
picture(ASSETS/'curves.png',34,101,892,292)
text(f'Mean fit time: CNN {summary.loc["cnn","train_seconds_mean"]:.2f} s; dropout {summary.loc["dropout","train_seconds_mean"]:.2f} s. Apple M4 CPU, four PyTorch threads.',48,399,865,size=19)
text('Online training uses changing weights and active dropout; validation uses fixed epoch-end weights. Their gap is descriptive.',48,447,865,size=18,color=MUTED)
foot('CNN: one setting per variant, three repeats. Baseline: three final candidates plus three repair fits. Sequential run times also reflect system load.')
end()
# 11
begin('Shirt and similar upper-body garments remain hardest')
picture(ASSETS/'classes.png',35,106,890,303)
shirt=classes[classes.class_name=='Shirt'].groupby('variant').f1.mean()
text(f'Shirt mean F1: CNN {pct(shirt["cnn"])}; dropout {pct(shirt["dropout"])}. Coat, Pullover and T-shirt/top follow.',48,421,865,size=22)
foot('Exploratory class-level analysis. Means and sample SD across seeds. Dropout improves some footwear/Bag categories, not every class.')
end()
# 12
begin('Actual confusion pairs: dropout, seed 11')
picture(ASSETS/'confusion.png',37,106,390,382)
text('Shirt is often predicted as:',466,119,440,size=24,bold=True)
lines([f'T-shirt/top: <b>{cm[6,0]}</b> images',f'Coat: <b>{cm[6,4]}</b> images',f'Pullover: <b>{cm[6,2]}</b> images'],x=466,y=168,w=440,size=25,gap=17)
text(f'T-shirt/top is predicted as Shirt <b>{cm[0,6]}</b> times. Coat is predicted as Pullover <b>{cm[4,2]}</b> times.',466,341,440,size=21)
text('Rows are true labels. Class IDs match slide 3.',466,435,440,size=17,color=MUTED)
foot('Exploratory error analysis. Seed 11 was designated before testing. Each row has 1,000 images; color shows the row fraction.')
end()
# 13
begin('Representative errors and visible similarities')
raw=read_idx(ROOT/'data/raw/t10k-images-idx3-ubyte.gz')
ids=[40,244,25,117]
examples={e['test_index']:e for e in errors['examples']}
notes={40:'Pale short sleeves and faint central detail.',244:'Long sleeves and a nearly uniform pale front.',25:'Dark long sleeves with weak visible front detail.',117:'Long sleeves, bright collar and central detail.'}
for j,index in enumerate(ids):
 e=examples[index];x=48+j*221
 text(f'True: {e["true_class"]}<br/>Predicted: {e["predicted_class"]}',x,112,210,size=17,bold=True)
 im=Image.fromarray(raw[index]).resize((560,560),resample=Image.Resampling.NEAREST)
 c.drawImage(ImageReader(im),x,H-177-184,width=184,height=184)
 text(f'Test index {index}',x,371,210,size=15,color=MUTED)
 text(notes[index],x,403,203,size=17)
foot('Exploratory visual descriptions, not causal feature attribution. Examples follow the saved deterministic error-pair selection rule.')
end()
# 14
begin('Code walkthrough and reproducibility')
lines(['<b>src/data.py · prepare</b><br/>Audit hashes, fix the split, fit normalization on training pixels.',
 '<b>src/study.py · SmallCNN / train_cnns</b><br/>Build the head, use a separate shuffle RNG, backpropagate, select by validation macro-F1.',
 '<b>freeze / evaluate / verify_saved</b><br/>Save provenance before testing, predict identical indices, reproduce metrics and checkpoint predictions.'],size=21,gap=20)
text('.venv/bin/python scripts/run_notebook.py --mode train',48,411,865,size=19,color=TEAL)
text('Quick check: .venv/bin/python scripts/verify_package.py',48,450,865,size=18,color=MUTED)
foot('The clean-kernel training notebook completed all 8 code cells with no errors. Executed notebook, versions and results are in the ZIP.')
end()
# 15
begin('Conclusion and limitations')
text(f'<b>Measured answer:</b> head dropout reduced mean test macro-F1 by {abs(mean_delta):.3f} percentage points. All three paired differences were negative.',48,119,865,size=27,color=ORANGE)
lines(['Shirt and similar upper-body categories remain the main difficulty.',
 'One dataset, one fixed split, one dropout rate and a 12-epoch budget limit the scope of this finding.',
 'Extra regularization may slow fitting or be unnecessary at this capacity. These are possible explanations, not proven mechanisms.'],y=245,size=23,gap=22)
foot('No claim of population-level significance or performance on new camera images. Near-duplicate/acquisition dependence remains untested.')
end()
# 16
begin('Team contributions and recording completion')
lines(['<b>Team to complete:</b> actual member names and each person’s real contribution. Every member must speak and explain their work.',
 'The user reports team review of existing work. Check the final generated package together before recording and submission.',
 '<b>Still pending:</b> the 15-minute recording, evidence of every member’s participation, instructor viewing access and Moodle submission.'],size=23,gap=27)
text('Submission deadline: December 4, 2026, 23:59 Astana time (UTC+5).',48,436,865,size=20,color=TEAL)
foot('Speaking roles and exact timings: docs/recording_outline.md. No member identities or contributions have been invented.')
end()
# 17
begin('References and implementation credit')
refs=[
 '<b>Dataset:</b> H. Xiao, K. Rasul and R. Vollgraf, “Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms,” 2017. arxiv.org/abs/1708.07747v2',
 '<b>Dropout:</b> N. Srivastava et al., “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” JMLR 15, pp. 1929-1958, 2014. jmlr.org/papers/v15/srivastava14a.html',
 '<b>Training pattern:</b> PyTorch, Learn the Basics: Optimizing Model Parameters. docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html',
 '<b>Baseline:</b> scikit-learn LogisticRegression API. scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html']
lines(refs,y=112,size=18,gap=19)
text('Code prepared for this study with AI assistance. No pretrained model or external training data. Team members must review and understand the implementation.',48,421,865,size=18,color=MUTED)
foot('All experimental numbers come from the verified results/ files. Full citations and methodology are in the report.')
end()
c.save()
(ROOT/'docs/slides_index.md').write_text('# Slides index\n\n'+ '\n'.join(f'{i+1}. {t}' for i,t in enumerate(titles))+'\n\nBuild: `.venv/bin/python scripts/build_slides.py`. Edit `TEAM_LINE` and the contribution slide with actual member details before final submission.\n')
print(f'Built {OUT} ({len(titles)} slides)')
