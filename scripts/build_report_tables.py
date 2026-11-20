"""Generate report numbers and tables from verified saved results (no fitting)."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; O=ROOT/'report/generated'; O.mkdir(parents=True,exist_ok=True)
comparison=pd.read_csv(R/'comparison.csv')
summary=pd.read_csv(R/'summary.csv').set_index('variant')
pairs=pd.read_csv(R/'matched_seed_deltas.csv')
training=pd.read_csv(R/'training_runs.csv')
classes=pd.read_csv(R/'class_metrics.csv')
baseline=json.loads((R/'baseline_tuning.json').read_text())
audit=json.loads((R/'data_audit.json').read_text())
checks=json.loads((R/'checks.json').read_text())
dev=json.loads((R/'development_log.json').read_text())
macros={}
for variant,prefix in [('logistic','Log'),('cnn','Plain'),('dropout','Drop')]:
 for metric,suffix in [('accuracy','Acc'),('macro_f1','Fone')]:
  macros[prefix+suffix]=f"{100*summary.loc[variant,metric+'_mean']:.3f}"
  if variant!='logistic': macros[prefix+suffix+'SD']=f"{100*summary.loc[variant,metric+'_sd']:.3f}"
 macros[prefix+'Time']=f"{summary.loc[variant,'train_seconds_mean']:.2f}"
 if variant!='logistic': macros[prefix+'TimeSD']=f"{summary.loc[variant,'train_seconds_sd']:.2f}"
macros.update({'MeanDelta':f"{pairs.delta_macro_f1_percentage_points.mean():.3f}",
 'NeuralGain':f"{100*(summary.loc['cnn','macro_f1_mean']-summary.loc['logistic','macro_f1_mean']):.3f}",
 'BaselineBudget':f"{baseline['total_seconds']:.2f}",
 'BaselineAllBudget':f"{baseline['total_seconds']+sum(c['seconds'] for c in dev['initial_candidates']):.2f}",
 'NormMean':f"{audit['preprocessing']['mean']:.8f}",'NormSD':f"{audit['preprocessing']['std']:.8f}",
 'CheckInitialLoss':f"{checks['finite_loss']:.5f}",'CheckFinalLoss':f"{checks['subset_final_loss']:.6f}",
 'CNNParams':f"{checks['parameters']:,}",'LogParams':f"{baseline['parameters']:,}",
 'TotalCNNTime':f"{training.train_seconds.sum():.2f}"})
(O/'numbers.tex').write_text('%% Generated from verified results. Do not hand-edit.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
rows=[]
for r in comparison.to_dict('records'):
 name='Logistic' if r['variant']=='logistic' else ('CNN' if r['variant']=='cnn' else 'CNN + dropout')
 seed='--' if r['variant']=='logistic' else str(int(r['seed']))
 epoch='--' if r['variant']=='logistic' else str(int(r['selected_epoch']))
 rows.append(f"{name} & {seed} & {epoch} & {r['accuracy']*100:.3f} & {r['macro_f1']*100:.3f} & {r['train_seconds']:.2f} \\\\")
(O/'run_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for r in pairs.to_dict('records'):
 rows.append(f"{int(r['seed'])} & {100*r['cnn_macro_f1']:.3f} & {100*r['dropout_macro_f1']:.3f} & {r['delta_macro_f1_percentage_points']:+.3f} \\\\")
(O/'paired_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for r in baseline['candidates']:
 rows.append(f"{r['C']:g} & {r['iterations']} & {100*r['macro_f1']:.3f} & {r['seconds']:.2f} \\\\")
(O/'baseline_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for label in range(10):
 group=classes[classes.class_id==label]
 name=group.iloc[0].class_name
 vals=group.groupby('variant').f1.mean()
 rows.append(f"{label} & {name} & {100*vals['logistic']:.2f} & {100*vals['cnn']:.2f} & {100*vals['dropout']:.2f} \\\\")
(O/'class_rows.tex').write_text('\n'.join(rows)+'\n')
print('Generated',len(list(O.glob('*.tex'))),'report fragments from verified CSV/JSON.')
