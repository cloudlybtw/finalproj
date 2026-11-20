# ROBT407 mandatory requirements checklist

Source of truth: `ROBT407_Final_Project_2026.pdf`, read in full (five pages). Page numbers below are the printed assignment page numbers, which match PDF pages. This is a requirements and evidence index, not a grade claim. An unchecked box means completion has not yet been verified in this index.

## Study and experimental design

| ID | Mandatory requirement | Assignment source | Verification status and evidence |
|---|---|---|---|
| S01 | [x] Define a feasible supervised task and explain how the model, data, and evaluation support the conclusions. | p. 1, §§1–2 | **PASS** — Notebook cells 0–1 state ten-class clothing task, research question and predicted effect; results cells 12/14 answer the comparison; slides 2 and 15. |
| S02 | [x] Cite the original dataset and document how to obtain it. | p. 1, §2; p. 3, §7 | **PASS** — Notebook §2/cell 3 credits Xiao/Rasul/Vollgraf; `README.md` Data and split; `scripts/download_data.py`; author file hashes in `src/data.py`; slides 3 and 17. |
| S03 | [x] Choose sample subset, classes, and inclusion/exclusion rules before comparing performance. | p. 1, §2; p. 2, §4 | **PASS** — Pre-run notebook §1/cell 1 and `config.json` specify all ten classes, inclusion and duplicate policy; `results/data_audit.json` records zero exclusions. |
| S04 | [x] Train a credible classical non-neural baseline appropriate to the task. | p. 1, §3.1 | **PASS** — `src/study.py:Study.baseline`; notebook §4/cell 8; three converged final candidates in `results/baseline_tuning.json`; selected C = 0.01; slide 5. |
| S05 | [x] Implement and train a suitable PyTorch neural model; merely calling a pretrained model is insufficient. | p. 2, §3.2 and following paragraph | **PASS** — `src/study.py:SmallCNN` and `Study.train_cnns`; notebook §3/§5, cells 6/10; six saved state dictionaries and all histories. Random initialization, no pretraining. |
| S06 | [x] Change exactly one aspect of the neural pipeline and compare original and modified models consistently. | p. 2, §3.3 and §4 | **PASS** — `SmallCNN(dropout)` changes only the head's dropout probability; `config.json`; equal parameter counts in `results/checks.json`; slides 6–7. |
| S07 | [x] Write the prediction and rationale in the notebook before running the controlled experiment. | p. 2, §4 | **PASS** — Notebook §1/cell 1 precedes every code cell and states p = 0.30, predicted improvement and rationale; training execution metadata is retained. |
| S08 | [x] State what changes and what remains fixed; keep partitions, inclusion rules, primary metric, and tuning budgets comparable. | p. 2, §4 | **PASS** — Notebook §1 fixed protocol; config; same split file, normalization, Adam/lr/batch/epochs/selection; independent shuffle RNG in `Study.train_cnns`; slide 7. |
| S09 | [x] Explain architecture, tensor dimensions, parameter count, loss, optimizer, and training loop. Cite reused code and papers. | p. 2, paragraph after §3.3; p. 4, §8 | **PASS** — Notebook §3/cell 6 prints architecture, shapes, loss and checks; `results/checks.json`; `docs/code_provenance.md`; report pp. 2–3/7; slides 6–7/17. |
| S10 | N/A: If using pretrained weights, disclose their origin, pretraining data, frozen/trainable layers, fine-tuning, and advantages. | p. 2, §4 | **N/A (condition checked)** — Both CNNs are initialized randomly; no pretrained weights or external training examples. Notebook §1, README provenance, slide 6. |

## Data handling and evaluation

| ID | Mandatory requirement | Assignment source | Verification status and evidence |
|---|---|---|---|
| D01 | [x] Define the target and partition first; record dataset version, counts, class balance, missing data, and duplicates. | p. 2, §5.1 | **PASS** — `results/data_audit.json` contains release, shape/dtype/range, counts, class balance, missingness, labels and duplicate audit; notebook §2/cell 4; slides 3–4. |
| D02 | [x] Retain a suitable official test set; otherwise justify the test split. | p. 2, §5.1 | **PASS** — All 10,000 official labels/indices independently verified against original IDX; `results/split_indices.npz` and `test_predictions.npz`; slide 4. |
| D03 | [x] Respect dependence: keep related acquisitions and duplicate/augmented versions together; avoid partition leakage. | p. 2, §5.2 | **PASS with stated scope limitation** — `src/data.py:prepare` excludes overlapping/conflicting hashes and deduplicates before splitting; audit reports zero exact overlaps. No group IDs exist, and near-duplicate/acquisition dependence is explicitly acknowledged in audit/README/slide 4. |
| D04 | [x] Fit learned preprocessing on training data only and apply it to validation/test; augmentation, if used, is training-only. | p. 2, §5.3 | **PASS** — `src/data.py:prepare` calculates mean/SD only on training indices; `src/study.py:tensor` applies them unchanged; independent recomputation matches. No augmentation, imputation, or feature fitting outside training. |
| D05 | [x] Tune hyperparameters, stopping, and checkpoints using validation data without consulting test scores. Use the same validation design across models. | p. 2, §5.4 | **PASS** — Baseline validation macro-F1 chooses C; CNN histories choose highest validation macro-F1/earliest tie; independently checked all six selected epochs; notebook §§4–5/cells 8/10. |
| D06 | [x] Freeze the procedure before final test evaluation. | p. 2, §5.5 | **PASS** — Notebook §6/cell 12 calls freeze before evaluation; `results/frozen_protocol.json` contains settings/checkpoint/split/source SHA-256s and no-test-consulted record; hashes independently intact. |
| D07 | [x] Run each neural variant with three recorded seeds on the same fixed split; report every seed without selecting a best test run. | p. 2, §5.5 | **PASS** — Seeds 11/22/33 for each variant in `config.json`, `training_runs.csv`, `comparison.csv`; same fixed split; all six runs reported, slide 9. |
| D08 | [x] Evaluate the baseline and neural models on the same held-out examples. | p. 2, §5.5 | **PASS** — Seven saved prediction arrays share identical official indices/labels; all checkpoints independently reproduce them; comparison contains every run. |
| D09 | [x] Document reproduction choices and acknowledge that seeds do not ensure identical results across hardware/software. | p. 2, final paragraph of §5 | **PASS** — README reproduction commands, exact `requirements.txt`, `results/environment.json`, config/seeds, frozen hashes; README notes cross-hardware/software variation. |

## Required evidence and interpretation

| ID | Mandatory requirement | Assignment source | Verification status and evidence |
|---|---|---|---|
| E01 | [x] Include a dataset summary and partition table, with grouping/chronology if relevant. | p. 3, §6 | **PASS** — Notebook §2/cell 4 partition/class-balance output; `results/data_audit.json`; slide 4; report p. 1 Table I. |
| E02 | [x] Report classification confusion matrix, accuracy, macro-F1, label mapping, and primary metric; add balanced accuracy if classes are imbalanced. | p. 3, §6 | **PASS** — `comparison.csv` contains accuracy/macro-F1/balanced accuracy for all runs; all seven `confusion_*.csv`; notebook §7/cells 14–15; labels in audit and slide 3; primary metric stated in §1. |
| E03 | [x] Compare the classical baseline, original neural model, and controlled improvement; include individual seeds, means, and standard deviations. | p. 3, §6 | **PASS** — All seven rows, two CNN mean/SD summaries and three paired deltas in `comparison.csv`, `summary.csv`, `matched_seed_deltas.csv`; independently recomputed; slides 8–9. |
| E04 | [x] Interpret three seeds as training variability, not population-level statistical significance. | p. 3, §6 | **PASS** — Notebook §§1/8; README and recording outline; slide 9 explicitly distinguishes training variability from population significance. |
| E05 | [x] Include training/validation learning curves, parameter counts, training times, hardware, and tuning budget. | p. 3, §6 | **PASS** — Six histories, `figures/learning_curves.pdf`, parameter counts, `training_runs.csv`, hardware/version record, tuning JSON and pre-test repair log; slides 5–7/9–10. |
| E06 | [x] Perform at least one task-relevant error analysis; label interpretations formed after seeing test errors exploratory. | p. 3, §6 | **PASS, display-only repair verified separately** — `class_metrics.csv`, `error_pairs.csv`, `errors.json`; fixed illustrative seed 11, deterministic first-index examples, explicitly exploratory captions; slides 11–13. Representative-grid spacing repair does not change data. |
| E07 | [x] Explain possible causes and alternative explanations; explicitly state if the hypothesis is unsupported. | p. 3, final paragraph of §6 | **PASS** — Negative result stated directly, all three paired differences reported; alternative mechanisms and fixed-budget limitations are framed as possibilities; notebook §8, recording outline, slides 15. |

## Submission deliverables and team actions

| ID | Mandatory requirement | Assignment source | Verification status and evidence |
|---|---|---|---|
| P01 | [x] Report is 4–6 pages excluding references, compiled from standard two-column IEEEtran conference LaTeX with normal margins/fonts and no template placeholders. | p. 3, §7.1 | **PASS** — `report/report.tex` uses genuine unmodified `IEEEtran.cls` in conference mode. Revised report has 6 content pages plus references on p. 7. Every page inspected; no overfull boxes/undefined refs. `--mode` on p. 5 was repaired and checked; report hash matches the refreshed QA record. |
| P02 | [ ] List every team member in the report and include a contribution statement identifying each person's work. | p. 3, §7; p. 4, §8 | **PENDING TEAM** — Actual names and individual contribution details have not been supplied. Explicit completion notices appear in report, slide 1/16, and `docs/team_completion.md`; none have been invented. |
| P03 | [ ] Contribution statement confirms all members reviewed the full submission. | p. 4, §8 | **PARTIAL / PENDING FINAL REVIEW** — User reports review of existing work. The final report/slides/package are generated afterward, so final-artifact review confirmation remains pending; documented in team completion instructions and slide 16. |
| P04 | [x] Code ZIP includes executed notebooks with visible outputs, supporting modules, dependency/version file, configuration and seeds, saved numerical results, README reproduction instructions, and report LaTeX sources. | p. 3, §7.2 | **PASS (tested archive and full reproduction)** — Tested ZIP has all required notebook/code/dependency/config/result/report-source entries and report-generated assets; all 96 payload hashes verified. Full training from its extraction exactly reproduces all seven prediction vectors; `results/archive_reproduction.json`. |
| P05 | [x] Supply dataset download instructions instead of uploading large raw public datasets. | p. 3, §7.2 | **PASS** — ZIP inventory independently confirms raw `data/`, virtual environments and self-assessment are absent. README and `scripts/download_data.py` document checksum-verified downloads. |
| P06 | [x] Supply slides PDF with readable figures/captions and a clear main comparison. | p. 3, §7.3 | **PASS** — `output/pdf/slides.pdf` has 17 readable slides; main comparison on 8 and every seed on 9. All pages visually checked; `docs/slides_verification.md`. Numeric checks agree with saved data. |
| P07 | [ ] Supply a prerecorded 15-minute team video; every member speaks and explains their contribution. | p. 3, §7.4 | **PENDING TEAM** — No recording exists in observed evidence. `docs/recording_outline.md` provides timed speaking assignments for any actual team size 1–4, but an outline does not prove participation or fulfill video requirement. |
| P08 | [ ] Recording covers question, data, split, models, controlled change, results, limitations, and a short implementation code walkthrough. | p. 3, §7.4 | **PLAN COMPLETE / RECORDING PENDING** — Outline covers question/data/split/models/change/results/limits and a 90-second code walkthrough at 12:00–13:30. Actual recorded coverage still requires inspection. |
| P09 | [ ] Upload recording to Moodle if possible; otherwise provide a short PDF with a stable viewing link, accessible to the instructor without requesting access until grading ends. | p. 3, §7 | **PENDING TEAM** — No uploaded recording or stable viewing-link PDF/access test is observed. Team instructions specify instructor access without a request and retention through grading. |
| P10 | [ ] One designated team member submits the full package via Moodle by December 4, 2026, 23:59 Astana time (UTC+5). Teams have at most four students. | pp. 1, 3, §7 | **PENDING TEAM** — Actual roster/count and designated submitter are not supplied; Moodle receipt absent. Due December 4, 2026, 23:59 Astana time (UTC+5). |

## Additional verification explicitly requested for this project

These are user-requested requirements or implementation checks, distinguished from the assignment's mandatory deliverables. Assignment p. 5, §10.2 also recommends the batch and small-subset checks.

| ID | Requirement | Verification status and evidence |
|---|---|---|
| U01 | [x] Use all ten Fashion-MNIST classes and a single fixed stratified validation split. | **PASS** — All ten labels, 48k/12k stratified fixed split seed 407, official 10k test set; data audit and independently verified split. |
| U02 | [x] Choose and record dropout rate before comparison; change classification-head dropout only. | **PASS** — Pre-run notebook §1 p = 0.30; model signature makes head dropout the only structural pipeline change; same 105,866 parameters. |
| U03 | [x] Verify shapes, labels, finite loss, backward pass, and learning on a small training subset. | **PASS** — `results/checks.json`: input 64×1×28×28, logits 64×10, int64 labels 0–9, finite loss/gradients, 64-sample accuracy 1.0 after 180 steps. |
| U04 | [x] Execute the notebook from a clean kernel using README commands before drafting the report; retain visible outputs. | **PASS** — `scripts/run_notebook.py` launches a new kernel. `notebook_train_verification.json`: 8/8 code cells, zero errors, 96 outputs; visual refresh disclosed separately. |
| U05 | [x] Saved predictions reproduce accuracy and macro-F1 for all seven runs; include matched-seed macro-F1 changes. | **PASS** — Independent checkpoint/prediction/count audit, saved predictions, checkpoints and independent count-based checks; seven accuracy/macro-F1 values and three matched differences reproduce to 1e−12. |
| U06 | [x] Give a code checkpoint containing reproduction command, measured comparison, and check outcomes; proceed to report automatically. | **PASS (parent workflow record)** — The code checkpoint was provided before report drafting, including reproduction command, measured comparison and successful checks. Notebook completed before report/slides authoring. |
| U07 | [x] Inspect the compiled report PDF page by page and cross-check every reported number against verified outputs. | **PASS** — Numerical sources and final seven-page PDF checked; six content pages comply. All pages visually inspected. Report pp. 1/3–4/6 and slides 8–9/15 agree with numerical outputs; `scripts/crosscheck_artifacts.py` passes. Literal `--mode` repair on p. 5 is verified. |
| U08 | [x] Extract the code ZIP into a fresh directory and test documented instructions from that copy. | **PASS** — Fresh extracted copy, new virtual environment, 74 pinned dependencies, pip check, full clean-kernel training (8 cells/0 errors) and exact seven-run prediction/history reproduction; `results/archive_reproduction.json`. Extracted report compilation also passed. |
| U09 | [ ] Prepare a timed 15-minute recording outline, short code walkthrough, and speaking part for each real team member. | **OUTLINE COMPLETE / ACTUAL NAMES PENDING** — `docs/recording_outline.md`: 15-minute timings, adaptable assignments for 1–4 members, technical paths, truthful contributions; actual member mapping remains team work. |
| U10 | [x] Reopen the assignment and assess every requirement with specific evidence; fix recoverable gaps and reassess. | **PASS** — Assignment reopened and fully reread; every mandatory/user requirement mapped here. Figure spacing, report length, generated-asset packaging and literal command typography repaired and reassessed; full archive reproduction passes. Team-only obligations remain explicitly pending. |
| U11 | [ ] Finish with measured answer, deliverable links, current estimated score /100 with criterion evidence/deductions, and remaining team actions. | **READY FOR FINAL RESPONSE** — Measured answer and final deliverable paths are prepared; the final response will provide links, criterion-level self-assessment and remaining team actions. The self-assessment is excluded from the submission archive. |

## Exact Section 8 rubric

| Criterion (assignment wording) | Maximum |
|---|---:|
| Problem definition, dataset understanding, and hypothesis | 10 |
| Valid preprocessing, partitions, and model selection | 25 |
| Neural-network implementation and technical understanding | 20 |
| Credible baseline and controlled comparison | 20 |
| Results, error analysis, and interpretation | 15 |
| Code quality, documentation, and reproducibility | 5 |
| Recorded presentation and individual contribution | 5 |
| **Total** | **100** |

Source: assignment p. 4, §8. The schedule on p. 4, §9 is suggested, not an additional submission requirement. The 70/15/15 split, Adam learning rate 0.001, batch size 64, and 30-epoch example are starting suggestions, not mandatory settings. Fashion-MNIST is optional in the assignment but mandatory in this user's project. The random-initialization route makes the pretrained-weight disclosure condition in S10 inapplicable.
