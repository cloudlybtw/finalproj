"""Controlled Fashion-MNIST experiment. No test scores are read during fitting."""
import copy, hashlib, json, os, platform, subprocess, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
import joblib, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.data import prepare, save_json, LABELS

class SmallCNN(nn.Module):
 def __init__(self,dropout=0.):
  super().__init__()
  self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
  self.head=nn.Sequential(nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,10))
 def forward(self,x): return self.head(self.features(x))

def metrics(y,p):
 return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)), 'balanced_accuracy':float(balanced_accuracy_score(y,p))}

def seed_all(seed):
 import random
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def tensor(x,d): return torch.from_numpy((x.astype(np.float32)/255-d['mean'])/d['std']).unsqueeze(1)

@torch.inference_mode()
def predict(model,x,y=None,batch=512):
 model.eval(); out=[]; loss=0.
 for start in range(0,len(x),batch):
  logits=model(x[start:start+batch]); out.append(logits.argmax(1).numpy())
  if y is not None: loss+=nn.functional.cross_entropy(logits,y[start:start+batch],reduction='sum').item()
 return np.concatenate(out),loss/len(x)

class Study:
 def __init__(self,root=Path('.')):
  self.root=Path(root).resolve(); self.cfg=json.loads((self.root/'config.json').read_text())
  for p in ['results','checkpoints','figures']: (self.root/p).mkdir(exist_ok=True)
  torch.set_num_threads(self.cfg['threads']); torch.use_deterministic_algorithms(True)
  self.d=prepare(self.root,self.cfg)
  self.x=tensor(self.d['train_x'],self.d); self.y=torch.from_numpy(self.d['train_y'].astype(np.int64))
  self.vx=tensor(self.d['val_x'],self.d); self.vy=torch.from_numpy(self.d['val_y'].astype(np.int64))
 def hardware(self):
  hw={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),
      'torch_threads':torch.get_num_threads(),'device':'cpu','cuda_available':torch.cuda.is_available(),'mps_available':torch.backends.mps.is_available(),
      'versions':{k:version(k) for k in ['torch','scikit-learn','numpy','pandas','matplotlib','nbformat','nbclient','ipykernel']}}
  for key,cmd in [('cpu_model',['sysctl','-n','machdep.cpu.brand_string']),('memory_bytes',['sysctl','-n','hw.memsize'])]:
   try: hw[key]=subprocess.check_output(cmd,text=True,stderr=subprocess.DEVNULL).strip()
   except (FileNotFoundError,subprocess.CalledProcessError): hw[key]='unavailable'
  save_json(self.root/'results/environment.json',hw); return hw
 def checks(self):
  seed_all(407); m=SmallCNN(); xb,yb=self.x[:64],self.y[:64]
  shapes={}; handles=[]
  for name,module in m.named_modules():
   if name and not list(module.children()):
    handles.append(module.register_forward_hook(lambda mod,inp,out,n=name: shapes.setdefault(n,list(out.shape)) and None))
  out=m(xb)
  for h in handles: h.remove()
  assert out.shape==(64,10) and yb.dtype==torch.int64 and yb.min()>=0 and yb.max()<10
  loss=nn.functional.cross_entropy(out,yb); assert torch.isfinite(loss)
  loss.backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
  initial=float(loss.detach()); opt=torch.optim.Adam(m.parameters(),lr=self.cfg['learning_rate'])
  for _ in range(180):
   m.train(); opt.zero_grad(); l=nn.functional.cross_entropy(m(xb),yb); l.backward(); opt.step()
  pred,last=predict(m,xb,yb); acc=float(accuracy_score(yb,pred))
  assert acc>=0.95 and last<initial*.2,(acc,last,initial)
  seed_all(11); a=SmallCNN(0); seed_all(11); b=SmallCNN(self.cfg['dropout'])
  assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
  a.eval(); b.eval(); assert torch.equal(a(xb),b(xb))
  b.train(); p=b(xb); q=b(xb); assert not torch.equal(p,q)
  result={'status':'PASS','input_shape':list(xb.shape),'output_shape':list(out.shape),'labels':'int64, 0..9','finite_loss':initial,
   'finite_gradients':True,'subset_samples':64,'subset_steps':180,'subset_accuracy':acc,'subset_final_loss':last,
   'paired_initial_weights_equal':True,'dropout_off_in_eval':True,'dropout_stochastic_in_train':True,
   'parameters':sum(p.numel() for p in a.parameters()),'layer_shapes':shapes}
  save_json(self.root/'results/checks.json',result); return result
 def baseline(self):
  X=self.x.flatten(1).numpy(); V=self.vx.flatten(1).numpy(); rows=[]; best=-1; best_model=None
  for c in self.cfg['baseline_C']:
   model=LogisticRegression(C=c,solver='lbfgs',max_iter=self.cfg['baseline_max_iter'],tol=self.cfg['baseline_tol'])
   t=time.perf_counter()
   with threadpool_limits(limits=self.cfg['threads']),warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always'); model.fit(X,self.y.numpy())
   elapsed=time.perf_counter()-t; p=model.predict(V); met=metrics(self.vy,p)
   row={'C':c,'seconds':elapsed,'iterations':int(model.n_iter_.max()),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in ws),**met}
   rows.append(row); print('Baseline validation',row,flush=True)
   if met['macro_f1']>best: best=met['macro_f1']; best_model=model; selected=row
  assert all(r['converged'] for r in rows),'Baseline solver did not converge; resolve before freezing'
  joblib.dump(best_model,self.root/'checkpoints/logistic.joblib')
  save_json(self.root/'results/baseline_tuning.json',{'candidates':rows,'selected':selected,'parameters':int(best_model.coef_.size+best_model.intercept_.size),'total_seconds':sum(r['seconds'] for r in rows)})
  return pd.DataFrame(rows)
 def train_cnns(self):
  summaries=[]
  for variant,drop in [('cnn',0.),('dropout',self.cfg['dropout'])]:
   for seed in self.cfg['seeds']:
    seed_all(seed); model=SmallCNN(drop); opt=torch.optim.Adam(model.parameters(),lr=self.cfg['learning_rate'],weight_decay=self.cfg['weight_decay'])
    # Dedicated generator prevents dropout consuming the minibatch-order RNG.
    generator=torch.Generator().manual_seed(seed)
    best=-1.; rows=[]; start=time.perf_counter()
    for epoch in range(1,self.cfg['epochs']+1):
     model.train(); order=torch.randperm(len(self.x),generator=generator); total=0.; train_y=[]; train_p=[]
     for ix in order.split(self.cfg['batch_size']):
      opt.zero_grad(set_to_none=True); logits=model(self.x[ix]); loss=nn.functional.cross_entropy(logits,self.y[ix]); assert torch.isfinite(loss)
      loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
      train_y.extend(self.y[ix].tolist()); train_p.extend(logits.detach().argmax(1).tolist())
     pred,vloss=predict(model,self.vx,self.vy); vm=metrics(self.vy,pred)
     row={'epoch':epoch,'train_loss_online':total/len(self.x),'train_macro_f1_online':f1_score(train_y,train_p,average='macro'),'val_loss':vloss,'val_macro_f1':vm['macro_f1'],'val_accuracy':vm['accuracy']}
     rows.append(row)
     save_json(self.root/'results/progress.json',{'variant':variant,'seed':seed,**row})
     if vm['macro_f1']>best:
      best=vm['macro_f1']; best_epoch=epoch; state=copy.deepcopy(model.state_dict())
     print(f'{variant} seed={seed} epoch={epoch} train_loss={row["train_loss_online"]:.4f} val_f1={best if False else vm["macro_f1"]:.5f}',flush=True)
    elapsed=time.perf_counter()-start
    torch.save(state,self.root/f'checkpoints/{variant}_{seed}.pt')
    pd.DataFrame(rows).to_csv(self.root/f'results/history_{variant}_{seed}.csv',index=False)
    summaries.append({'variant':variant,'seed':seed,'dropout':drop,'selected_epoch':best_epoch,'val_macro_f1':best,'train_seconds':elapsed,'epochs':self.cfg['epochs'],'parameters':sum(p.numel() for p in model.parameters())})
  pd.DataFrame(summaries).to_csv(self.root/'results/training_runs.csv',index=False)
  return pd.DataFrame(summaries)
 def freeze(self):
  paths=[self.root/'config.json',self.root/'results/split_indices.npz']+sorted((self.root/'checkpoints').glob('*'))+sorted((self.root/'src').glob('*.py'))
  record={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'test_metrics_consulted':False,'config':self.cfg,
   'selected_baseline':json.loads((self.root/'results/baseline_tuning.json').read_text())['selected'],
   'selected_cnns':pd.read_csv(self.root/'results/training_runs.csv').to_dict('records'),
   'sha256':{str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
  save_json(self.root/'results/frozen_protocol.json',record); return record
 def evaluate(self):
  frozen=json.loads((self.root/'results/frozen_protocol.json').read_text())
  for rel,h in frozen['sha256'].items(): assert hashlib.sha256((self.root/rel).read_bytes()).hexdigest()==h,rel
  tx=tensor(self.d['test_x'],self.d); y=self.d['test_y']; saved={'test_index':np.arange(len(y)),'y_true':y}; rows=[]
  model=joblib.load(self.root/'checkpoints/logistic.joblib'); p=model.predict(tx.flatten(1).numpy()); saved['logistic']=p
  tuning=json.loads((self.root/'results/baseline_tuning.json').read_text())
  rows.append({'run':'logistic','variant':'logistic','seed':None,'selected_epoch':None,'train_seconds':tuning['selected']['seconds'],'parameters':tuning['parameters'],**metrics(y,p)})
  for row in pd.read_csv(self.root/'results/training_runs.csv').to_dict('records'):
   name=f'{row["variant"]}_{row["seed"]}'; m=SmallCNN(row['dropout']); m.load_state_dict(torch.load(self.root/f'checkpoints/{name}.pt',weights_only=True))
   p,_=predict(m,tx); saved[name]=p
   rows.append({'run':name,**{k:row[k] for k in ['variant','seed','selected_epoch','train_seconds','parameters']},**metrics(y,p)})
  np.savez_compressed(self.root/'results/test_predictions.npz',**saved)
  table=pd.DataFrame(rows); table.to_csv(self.root/'results/comparison.csv',index=False)
  return table

def verify_saved(root=Path('.'),check_checkpoints=False):
 root=Path(root)
 frozen=json.loads((root/'results/frozen_protocol.json').read_text())
 for rel,h in frozen['sha256'].items():
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h, f'Frozen provenance mismatch: {rel}'
 p=np.load(root/'results/test_predictions.npz'); table=pd.read_csv(root/'results/comparison.csv'); checks=[]
 assert np.array_equal(p['test_index'],np.arange(10000))
 for row in table.to_dict('records'):
  computed=metrics(p['y_true'],p[row['run']]); assert all(abs(computed[k]-row[k])<1e-12 for k in computed)
  checks.append(row['run'])
 assert len(checks)==7 and set(p['y_true'])==set(range(10))
 if check_checkpoints:
  study=Study(root); tx=tensor(study.d['test_x'],study.d)
  assert np.array_equal(study.d['test_y'],p['y_true'])
  for row in table.to_dict('records'):
   if row['variant']=='logistic': q=joblib.load(root/'checkpoints/logistic.joblib').predict(tx.flatten(1).numpy())
   else:
    m=SmallCNN(0 if row['variant']=='cnn' else study.cfg['dropout']); m.load_state_dict(torch.load(root/f'checkpoints/{row["run"]}.pt',weights_only=True)); q,_=predict(m,tx)
   assert np.array_equal(q,p[row['run']]),row['run']
 return {'status':'PASS','frozen_hashes_match':True,'prediction_rows':10000,'runs_recomputed':checks,'all_metrics_match_at_1e_12':True,'checkpoint_predictions_identical':check_checkpoints}
