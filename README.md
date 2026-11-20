# ROBT407: head dropout on Fashion-MNIST

This repository contains the complete experiment, an executed notebook with visible outputs, verified predictions, an IEEEtran report and slides. The assignment is `ROBT407_Final_Project_2026.pdf`; its requirement index is `docs/requirements_checklist.md`. Names and individual contribution details must be supplied by the team. The recording and Moodle submission are pending.

## Reproduce the experiment

Tested on Python 3.14, macOS arm64, CPU. Exact versions are in `requirements.txt`; hardware and major versions are also in `results/environment.json`. Use the same versions where wheels are available. Other platforms may need a compatible PyTorch build, and exact floating-point results can differ.

From the extracted repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/run_notebook.py --mode train
```

The final command executes `notebooks/fashion_mnist.ipynb` in a **new kernel**, using the current interpreter, and saves `notebooks/fashion_mnist.executed.ipynb`. It trains three logistic candidates and six CNNs, freezes the procedure, evaluates the official test set, checks saved predictions against checkpoints, and generates the figures/tables. It overwrites model/result files. Preserve the delivered results separately if comparing a rerun. Allow sufficient time for 72 CNN epochs on CPU. All training data are loaded into RAM; allow several GB. No GPU is required.

To verify archive integrity immediately after extraction, run `.venv/bin/python scripts/verify_package.py` after installing dependencies. This validates the ZIP manifest, independently recomputes metrics from confusion counts, and checks frozen experiment hashes. Run it before regenerating outputs; afterward use `--skip-manifest` because generated PDFs and execution timestamps change.

For a quick fresh-kernel verification of the supplied results without retraining:

```sh
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/run_notebook.py --mode verify --output notebooks/fashion_mnist.verified.ipynb
```

The supplied `fashion_mnist.executed.ipynb` retains the full training outputs; `fashion_mnist.verified.ipynb` records a fresh-kernel replay. A post-execution error-grid spacing repair is documented in `results/figure_layout_fix.json`. It changes no experiment source, model or metric; `scripts/refine_figures.py` applies the repair in every notebook reproduction.

This checks every saved checkpoint's predictions on all 10,000 test examples against the saved arrays, recomputes all accuracy/macro-F1/balanced-accuracy values, regenerates tables and plots, and saves visible notebook outputs. It does not claim a second independent training run.

## Data and split

The data are the original 2017 Fashion-MNIST IDX release by Han Xiao, Kashif Rasul and Roland Vollgraf. Source: <https://github.com/zalandoresearch/fashion-mnist>; paper: <https://arxiv.org/abs/1708.07747v2>. `src/data.py` downloads four gzip files over verified HTTPS from the authors' repository and validates the authors' MD5 values. Raw datasets are deliberately excluded from the ZIP. There is no TorchVision dependency because a local IDX parser reads the original bytes directly.

Keep the official 10,000-example test set intact. Before splitting development data, remove hashes overlapping test, remove conflicting-label development hash groups and keep one example per remaining exact image hash. Then stratify the retained development examples 80/20 using seed 407. Saved original indices: `results/split_indices.npz`. Full label mapping, exclusions, class balance and training-only global normalization: `results/data_audit.json`.

## Study design and files

The pre-run hypothesis and controlled settings are in notebook section 1 and `config.json`. Both CNNs use 12 epochs, Adam at 0.001, batch size 128, paired seeds 11/22/33 and highest validation macro-F1 checkpoint selection (earliest tie). Dropout p=0.30 is the sole neural pipeline change. A dedicated shuffle generator keeps each matched seed's minibatch order identical. A three-candidate logistic C search uses only validation data. No test-set tuning is permitted.

- `src/study.py`: model, checks, training, freeze, test evaluation, verification.
- `src/analysis.py`: metrics summaries and publication figures.
- `results/comparison.csv`: baseline plus all six neural runs.
- `results/summary.csv`, `matched_seed_deltas.csv`: seed summaries and paired effects.
- `results/test_predictions.npz`: official indices, labels, all seven prediction vectors.
- `results/frozen_protocol.json`: timestamp, selected models and SHA-256 provenance.
- `results/history_*.csv`: all training and validation curves.
- `results/baseline_tuning.json`, `training_runs.csv`: selection and timing budgets.
- `checkpoints/`: baseline estimator and six PyTorch state dictionaries. Load these only from this trusted project.
- `report/`: LaTeX source and genuine IEEEtran class.
- `output/pdf/`: compiled report and slides.
- `docs/`: assignment checklist, recording outline and team completion instructions.

## Completed archive reproduction check

The submission was extracted into a fresh temporary project and all 74 pinned packages installed in a fresh virtual environment. Both the dependency check and archive-manifest check passed. The full training command then refitted all three baseline candidates and all six CNNs. Every test prediction vector, selected epoch, metric and learning-curve value exactly matched the delivered reference run. Times differed and are kept separate from the study's original timing results.

Evidence: `results/archive_reproduction.json`, `verification/fashion_mnist.retrained.ipynb` and `verification/retrained_comparison.csv`. The copied report sources also compiled successfully using archived inputs and the existing compiler/cache dependencies; `verification/archive_report_verification.json` records that check. No raw dataset files are packaged. The archive manifest protects the delivered files; regenerating outputs changes their hashes as expected.

## Report compilation

```sh
.venv/bin/python scripts/build_report_tables.py
.venv/bin/python scripts/build_report_figures.py
python3 tools/setup_tectonic.py
sh tools/compile_latex.sh report/report.tex output/pdf
```

First compilation downloads TeX support files; internet access is required. Tool provenance and cross-platform setup: `tools/LATEX_TOOLCHAIN.md`. Alternatively use a standard TeX installation with IEEEtran and compile from `report/`. The report tables are generated from saved results as documented in the report sources. The slides are regenerated with `.venv/bin/python scripts/build_slides.py`.

## Sources and credit

The local code was written for this study with AI assistance; no pretrained model or external training data are used. The training loop follows the standard PyTorch zero-grad / forward / cross-entropy / backward / optimizer-step pattern described in <https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>. Logistic regression and metrics use scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>. Dropout: Srivastava et al. (2014), <https://jmlr.org/papers/v15/srivastava14a.html>. The report cites these sources. Team members must understand the code and replace the team metadata before submitting.
