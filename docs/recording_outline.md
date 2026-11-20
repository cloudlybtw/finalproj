# Timed team recording outline (15 minutes)

Use the finished slides PDF and a readable editor/notebook window. This outline prepares the required recording; it is not evidence that a recording or individual participation has occurred. The user reports that all members reviewed the work and will supply actual names and contributions. The final generated artifacts still need a team check before recording. Do not invent names, membership count, or work performed.

## Assign actual speakers before rehearsal

The letters below are speaking roles, not claimed team members. Replace them with actual names. Choose the row matching the real team size (one to four students). Each person must speak and explain their own contribution; distributing speaking roles does not establish who performed the work.

| Actual team size | Suggested assignment |
|---|---|
| 1 | One person presents all sections and their own contribution. |
| 2 | A: 0:00–5:00 and 12:00–13:30. B: 5:00–12:00 and 13:30–14:00. Both: 14:00–15:00, about 30 seconds each. |
| 3 | A: 0:00–2:00 and 5:00–8:00. B: 2:00–5:00 and 12:00–13:30. C: 8:00–12:00 and 13:30–14:00. All: 14:00–15:00, about 20 seconds each. |
| 4 | A: 0:00–2:00 and 13:30–14:00. B: 2:00–5:00. C: 5:00–8:00 and 12:00–13:30. D: 8:00–12:00. All: 14:00–15:00, about 15 seconds each. |

Record the chosen mapping in the team's preparation copy. Say names at each handoff, keep the cursor still while explaining, and use font sizes readable in the final recording. Rehearse to approximately 15:00, including transitions.

## 0:00–2:00 — Problem, motivation, and hypothesis

| Time | Talking points and on-screen evidence |
|---|---|
| 0:00–0:35 | Introduce the actual team and task: classify 28 × 28 grayscale clothing images into all ten Fashion-MNIST categories. State the question: does adding classification-head dropout to a small CNN improve held-out macro-F1, and which clothing categories remain hardest to distinguish? |
| 0:35–1:20 | Explain the prediction recorded before training: dropout at p = 0.30 should improve mean test macro-F1 by reducing dependence on particular hidden activations. This is a hypothesis, not a guarantee. Show notebook §1 and `config.json`. |
| 1:20–2:00 | Preview the controlled comparison: logistic regression, CNN, and identical CNN plus dropout. Primary metric is macro-F1, the unweighted mean of ten class F1 scores. Briefly disclose the measured answer: dropout did not improve it under this procedure. |

## 2:00–5:00 — Dataset, preprocessing, and evaluation design

| Time | Talking points and on-screen evidence |
|---|---|
| 2:00–2:45 | Credit Xiao, Rasul, and Vollgraf's Fashion-MNIST dataset (original 2017 IDX release). Explain that `scripts/download_data.py` obtains the four public files and verifies author-provided MD5 values. Labels 0–9 are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. Show the report/slide label mapping. |
| 2:45–3:35 | Explain the fixed stratified split: 48,000 training, 12,000 validation, and the unchanged official 10,000 test examples. Split seed is 407. Every class contributes 4,800/1,200/1,000 images. The data audit found no duplicate excess, cross-partition duplicate hashes, missing pixels, or missing labels; no images were excluded. Show `results/data_audit.json`. |
| 3:35–4:10 | Explain the predeclared duplicate policy: preserve test; exclude matching development hashes and conflicting-label groups; retain one development image per other hash before splitting. The zero-exclusion finding is a measured result of this rule. No acquisition/object identifiers were available, so exact matching cannot rule out near-duplicate or acquisition dependence. |
| 4:10–4:40 | Explain training-only normalization: divide pixels by 255, subtract training mean 0.285887, divide by training SD 0.352935. Apply those same constants to validation and test. Flatten only for logistic regression; keep image geometry for the CNN. |
| 4:40–5:00 | Validation selects settings/checkpoints; freeze the procedure before computing test metrics. All seven runs use the same 10,000 held-out examples. Balanced accuracy is also saved, although the test classes are balanced. |

## 5:00–8:00 — Models and the single controlled change

| Time | Talking points and on-screen evidence |
|---|---|
| 5:00–5:45 | Baseline: multinomial logistic regression, L2 regularization, 784 normalized pixel features, 7,850 coefficients/intercepts. Validation compares C = 0.01, 0.1, 1.0 and selects C = 0.01. All final candidates converged. Explain that a common iteration cap was raised from 500 to 5,000 before any CNN or test run to repair nonconvergence; this is logged in `results/development_log.json`. |
| 5:45–6:35 | Walk through the architecture: N × 1 × 28 × 28 input; 3 × 3 padded convolution to 16 channels, ReLU and 2 × 2 pooling; another convolution to 32 channels, ReLU and pooling; N × 32 × 7 × 7; flatten to 1,568; dense 64, ReLU, dropout, dense 10 raw logits. Both variants have 105,866 trainable parameters. |
| 6:35–7:20 | Only dropout changes: p = 0 versus 0.30 between hidden ReLU and final classifier. Both start randomly, use Adam at learning rate 0.001, zero weight decay, batch size 128, 12 epochs, and seeds 11, 22, 33. Dedicated shuffle generators keep matched-seed minibatch order identical. Highest validation macro-F1 selects the checkpoint, with earliest-epoch tie breaking. |
| 7:20–8:00 | Explain the loop: zero gradients, forward to raw logits, cross-entropy with int64 class labels, backward, optimizer update. Validation/test use evaluation mode and disabled gradients; dropout is then inactive. Checks passed for shapes, finite loss/gradients, paired initial weights, and dropout behavior. A 64-image subset reached 100% training accuracy after 180 steps. This is an implementation check, not held-out performance. |

## 8:00–12:00 — Results, learning curves, and exploratory errors

| Time | Talking points and on-screen evidence |
|---|---|
| 8:00–8:55 | Show the main comparison. Logistic regression: accuracy 84.52%, macro-F1 0.844119. CNN mean accuracy 90.98%; mean macro-F1 0.909274, sample SD 0.000167. Dropout mean accuracy 90.7933%; mean macro-F1 0.907063, sample SD 0.001457. The mean dropout effect is −0.002211 macro-F1, or −0.2211 percentage points. Avoid calling this a 0.2211 percent relative decrease. |
| 8:55–9:30 | Show all seed pairs. Macro-F1 changes (dropout minus CNN) are −0.3221, −0.0732, and −0.2679 percentage points for seeds 11, 22, and 33. The direction is negative for all three. Three seeds describe training variability on one split; they do not establish population-level significance. |
| 9:30–10:15 | Show learning curves for every run. Training loss falls; validation improvements become smaller toward the end. Online training metrics use evolving weights and active dropout; validation uses a fixed epoch-end model in eval mode, so their difference is not an exact generalization-gap estimate. The CNN selects epoch 12 for all seeds; dropout selects epochs 9, 12, 12. Mean fit times were 62.55 s for CNN and 77.43 s for dropout on an Apple M4 CPU using four PyTorch threads; these are observed timings, not universal speed claims. |
| 10:15–11:00 | Show per-class F1 and the seed-11 confusion matrix. Shirt is hardest: mean F1 0.734994 without dropout and 0.721451 with dropout. The next difficult categories are Coat (0.854066/0.847961), Pullover (0.859135/0.857524), and T-shirt/top (0.860289/0.857514). Trouser, footwear, and Bag are easier. Do not imply every category became worse: dropout improves mean F1 for Sandal, Sneaker, Bag, and Ankle boot in these runs. |
| 11:00–12:00 | Label the following interpretation **exploratory after inspecting test errors**. In dropout seed 11, 142 shirts are predicted as T-shirt/top, 73 as Coat, and 61 as Pullover; 74 T-shirts are predicted as Shirt. Show representative image indices 40 and 244: index 40 has a short-sleeved, light torso silhouette with faint central detail; index 244 has long sleeves and a mostly uniform pale front. These appearances share cues with the predicted categories. Index 25 has a dark long-sleeved outline with weak visible front detail, which may contribute to confusion between Coat and Pullover. These observations do not prove which pixels caused the model's decisions. Image examples are the first index for each of eight most frequent directed errors, not a random prevalence sample. Seed 11 was designated before testing. |

## 12:00–15:00 — Code walkthrough, limitations, conclusion, contributions

| Time | Screen/action and talking points |
|---|---|
| 12:00–12:25 | Show `src/data.py`, `prepare`: exact hashes and inclusion rules, fixed stratified split, train-only mean/SD, and `split_indices.npz`. Point out the explicit no-overlap assertion. |
| 12:25–12:55 | Show `src/study.py`, `SmallCNN` and `Study.train_cnns`: `nn.Dropout(dropout)`, shared architecture, separate minibatch RNG, raw-logit cross-entropy, optimizer update, and validation checkpoint selection. Explain one forward/backward step rather than scrolling through the entire module. |
| 12:55–13:15 | Show `Study.freeze`, `Study.evaluate`, and `verify_saved`: saved hashes/checkpoints before test evaluation, identical test indices for all models, and independent recomputation from prediction vectors. Show notebook §§6–7 with retained outputs. |
| 13:15–13:30 | Show the README reproduction commands and completed verification output. The clean-kernel training notebook executed all 8 code cells with 0 errors and 96 retained outputs. For a short recorded demonstration, run `.venv/bin/python scripts/verify_package.py` from the extracted project root after package preparation, or display its successful output. Do not spend the recording retraining the experiment. |
| 13:30–14:00 | State limitations and answer: one dataset, one fixed split, three seeds, one dropout rate, and a fixed 12-epoch budget limit generality. Extra regularization could slow fitting or be unnecessary at this capacity, but these are possible explanations rather than established mechanisms. The predicted macro-F1 improvement was not supported. Shirt and visually similar upper-body categories remain the main difficulty. |
| 14:00–15:00 | Every actual member gives their name, their actual contribution, and one technical detail they can explain. Allocate 60/N seconds to each member for N actual members. Use truthful wording such as “I implemented/reviewed/checked …” only for work actually performed. Confirm final team review only after it occurs. End on the measured finding, without claiming recording access or Moodle submission is already complete. |

## Numerical source map

- `results/summary.csv`: means, sample SDs, parameter counts and mean training times.
- `results/comparison.csv`: all seven individual runs and selected epochs.
- `results/matched_seed_deltas.csv`: signed paired macro-F1 differences.
- `results/class_metrics.csv`: per-run/per-class F1; class values above average the three CNN seeds.
- `results/confusion_dropout_11.csv`, `results/error_pairs.csv`, `results/errors.json`: counts and deterministic representative-image selection.
- `results/data_audit.json`, `baseline_tuning.json`, `checks.json`, `environment.json`, `notebook_train_verification.json`: design, checks, timings, hardware, and execution evidence.

## Recording and submission completion

After recording, check actual duration, sound, legibility, every member's speaking part, and the code walkthrough. Keep the video, individual participation, instructor access check, and Moodle submission pending until there is evidence. Upload the video if Moodle accepts its size; otherwise provide a short PDF with a stable viewing link that the instructor can open without requesting access. Keep access available until grading is complete. The deadline is **December 4, 2026, 23:59 Astana time (UTC+5)**. One designated actual member submits the complete package and retains the receipt.
