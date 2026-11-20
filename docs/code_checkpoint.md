# Verified code checkpoint

Reproduction: `.venv/bin/python scripts/run_notebook.py --mode train` (after README environment/data setup).

| Model | Test accuracy (%) | Test macro-F1 (%) |
|---|---:|---:|
| Logistic regression | 84.520 | 84.412 |
| CNN, mean ± sample SD | 90.980 ± 0.046 | 90.927 ± 0.017 |
| CNN + dropout, mean ± sample SD | 90.793 ± 0.153 | 90.706 ± 0.146 |

Paired macro-F1 changes for seeds 11/22/33: -0.322, -0.073, -0.268 percentage points. Mean change: -0.221 percentage points. The improvement hypothesis is unsupported under this fixed procedure.

Checks passed: original dataset checksums; fixed disjoint hash-audited partitions; tensor shapes and label types/ranges; finite loss and gradients; 64-image subset learning (100% accuracy); paired initial weights; dropout train/eval behavior; all baseline candidate convergence; frozen source/checkpoint hashes; all seven saved prediction vectors match checkpoint predictions on 10,000 official test examples; independent confusion-count reconstruction matches reported metrics.

The full clean-kernel training notebook completed 8/8 code cells with 96 retained outputs and no errors. A separate clean-kernel verification replay completed 8/8 code cells with 21 outputs and no errors. A visual-only image-grid spacing repair is documented in `results/figure_layout_fix.json`; no frozen experiment code or numerical result changed. The pre-test logistic solver-cap repair is documented in `results/development_log.json`.
