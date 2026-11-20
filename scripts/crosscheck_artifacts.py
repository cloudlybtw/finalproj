"""Check key reported values against saved metrics after final PDF compilation."""
from pathlib import Path
import json, re, hashlib
import pandas as pd
import pymupdf
root=Path(__file__).resolve().parents[1]
rows=pd.read_csv(root/'results/comparison.csv'); summary=pd.read_csv(root/'results/summary.csv').set_index('variant'); pairs=pd.read_csv(root/'results/matched_seed_deltas.csv')
checks={}
for name in ['report','slides']:
 doc=pymupdf.open(root/f'output/pdf/{name}.pdf')
 text='\n'.join(p.get_text() for p in doc).replace('−','-').replace('–','-')
 expected=[f'{100*v:.3f}' for v in rows.macro_f1]
 expected += [f'{100*summary.loc[v,k]:.3f}' for v in ['cnn','dropout'] for k in ['macro_f1_mean','macro_f1_sd','accuracy_mean','accuracy_sd']]
 expected += [f'{v:.3f}' for v in pairs.delta_macro_f1_percentage_points]
 expected += [f'{pairs.delta_macro_f1_percentage_points.mean():.3f}','105,866','7,850']
 normalized=re.sub(r'\s+','',text)
 for value in expected: assert value in normalized, f'{name}: missing expected numeric token {value}'
 if name=='report':
  assert '--modetrain' in normalized, 'Report must retain the literal double-hyphen command'
  references=next(i for i,p in enumerate(doc) if p.get_text().startswith('REFERENCES'))
  assert 4<=references<=6, f'Content pages {references}'
  checks['report_content_pages']=references
 checks[name]={'pages':len(doc),'expected_numeric_tokens_checked':len(expected),'sha256':hashlib.sha256((root/f'output/pdf/{name}.pdf').read_bytes()).hexdigest()}
train=json.loads((root/'results/notebook_train_verification.json').read_text()); fix=json.loads((root/'results/figure_layout_fix.json').read_text()); verify=json.loads((root/'results/notebook_verify_verification.json').read_text())
assert train['sha256']==fix['training_notebook_original_sha256']
assert hashlib.sha256((root/train['notebook']).read_bytes()).hexdigest()==fix['training_notebook_after_visual_refresh_sha256']
assert hashlib.sha256((root/verify['notebook']).read_bytes()).hexdigest()==verify['sha256']
checks['notebook_provenance']='Original full training + documented visual refresh + clean-kernel verification checked'
checks['status']='PASS'
(root/'results/artifact_crosscheck.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
