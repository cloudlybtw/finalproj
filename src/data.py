"""Original Fashion-MNIST IDX reader; checksums from the dataset authors."""
import gzip, hashlib, json, struct, urllib.request, ssl
import certifi
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

LABELS=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
FILES={
 'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
 'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
 't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
 't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
BASE='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'

def save_json(path,obj):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def download(root=Path('.')):
 raw=root/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
 for name,md5 in FILES.items():
  p=raw/name
  if not p.exists():
   print('Downloading',name,flush=True)
   with urllib.request.urlopen(BASE+name, context=ssl.create_default_context(cafile=certifi.where()), timeout=180) as response:
    p.write_bytes(response.read())
  assert hashlib.md5(p.read_bytes()).hexdigest()==md5, f'Checksum failure: {name}'
 return raw

def read_idx(path):
 b=gzip.decompress(path.read_bytes()); magic=struct.unpack('>I',b[:4])[0]
 if magic==2051:
  n,h,w=struct.unpack('>III',b[4:16]); a=np.frombuffer(b[16:],dtype=np.uint8).reshape(n,h,w)
 else:
  assert magic==2049
  n=struct.unpack('>I',b[4:8])[0]; a=np.frombuffer(b[8:],dtype=np.uint8).reshape(n)
 return a.copy()

def hashes(x):
 return np.array([hashlib.sha256(a.tobytes()).hexdigest() for a in x])

def prepare(root,cfg):
 raw=download(root)
 x=read_idx(raw/'train-images-idx3-ubyte.gz'); y=read_idx(raw/'train-labels-idx1-ubyte.gz')
 xt=read_idx(raw/'t10k-images-idx3-ubyte.gz'); yt=read_idx(raw/'t10k-labels-idx1-ubyte.gz')
 assert x.shape==(60000,28,28) and xt.shape==(10000,28,28)
 assert set(y)==set(yt)==set(range(10))
 hx,ht=hashes(x),hashes(xt); test_hashes=set(ht)
 groups={}
 for i,h in enumerate(hx): groups.setdefault(h,[]).append(i)
 conflicts={h for h,g in groups.items() if len(set(y[g]))>1}
 eligible=np.array([g[0] for h,g in groups.items() if h not in test_hashes and h not in conflicts])
 tr,va=train_test_split(eligible,test_size=cfg['validation_fraction'],random_state=cfg['split_seed'],stratify=y[eligible])
 tr,va=np.sort(tr),np.sort(va)
 assert not(set(hx[tr])&set(hx[va]) or set(hx[tr])&test_hashes or set(hx[va])&test_hashes)
 mean=float((x[tr].astype(np.float64)/255).mean()); std=float((x[tr].astype(np.float64)/255).std())
 np.savez_compressed(root/'results/split_indices.npz',train=tr,validation=va,test=np.arange(len(yt)))
 audit={'dataset':'Fashion-MNIST 2017 original release','source':BASE,'md5':FILES,
 'labels':dict(enumerate(LABELS)),'original_train':len(y),'official_test':len(yt),
 'counts':{'train':len(tr),'validation':len(va),'test':len(yt)},
 'class_counts':{k:np.bincount(v,minlength=10).tolist() for k,v in [('original_train',y),('train',y[tr]),('validation',y[va]),('test',yt)]},
 'missing_pixels':0,'missing_labels':0,'pixel_dtype':str(x.dtype),'pixel_range':[int(x.min()),int(x.max())],
 'duplicates':{'train_duplicate_excess':len(x)-len(groups),'test_duplicate_excess':len(xt)-len(set(ht)),
 'train_test_shared_hashes':len(set(hx)&test_hashes),'development_rows_matching_test':int(np.isin(hx,ht).sum()),
 'development_conflicting_label_groups':len(conflicts),'excluded_total':len(x)-len(eligible),
 'remaining_cross_partition_hash_overlap':0},'policy':cfg['duplicate_policy'],
 'preprocessing':{'mean':mean,'std':std,'fit_partition':'train only','operation':'(uint8/255 - mean)/std'},
 'split_seed':cfg['split_seed'],'grouping_limit':'No object/person IDs supplied; exact pixel hashes only. Near duplicates and unseen acquisition dependence may remain.'}
 save_json(root/'results/data_audit.json',audit)
 return {'train_x':x[tr],'train_y':y[tr],'val_x':x[va],'val_y':y[va],'test_x':xt,'test_y':yt,'mean':mean,'std':std,'audit':audit}
