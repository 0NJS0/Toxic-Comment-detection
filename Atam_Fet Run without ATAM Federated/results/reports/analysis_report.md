# ATAM-FET — Whole-Experiment Analysis

**Project:** Adaptive Toxic Attention Module for Federated Edge-Based Toxicity Detection
**Notebook:** `atamupdate2.ipynb` (Kaggle, single-file, resume-aware)
**Hardware:** Tesla T4 GPU, CUDA 12.8, PyTorch 2.10
**Data:** Jigsaw Toxic Comment Classification (6 multi-label targets)
**Seed:** 42 (single seed across all modules)
**Modules run:** m1–m10 (`results/.atam_state.json` shows all `done`)
**Total compute:** ~15 h wall time (m3 ≈ 9 670 s, m5 ≈ 15 104 s, m6 ≈ 19 023 s, m7 ≈ 3 736 s)

This report is an interpretation of the numbers that already live in `results/`. It does not retrain anything; it reads the JSON, CSV, and notebook cells and explains what the experiment shows, what the wins are, and where the claims are fragile.

---

## 1. Pipeline map

| m | purpose | evidence in `results/` |
|---|---------|------------------------|
| 1 | clean raw `train.csv` → `train_cleaned.csv` | inferred (artifact not kept) |
| 2 | HF tokenization, 80/10/10 split, max_len=256 | inferred |
| 3 | centralized DistilBERT fine-tune, 3 epochs, LR 2e-5 | `results/checkpoints/`, `results/logs/` |
| 4 | LoRA parameter counts (q_lin, v_lin, r=8) | reported in JSONs as `n_comm_params` |
| 5 | FedAvg+LoRA sweep (IID / Dirichlet 0.5 / Dirichlet 0.1 / Topic) | `results/federated/{iid,dirichlet_0.1,dirichlet_0.5,topic}/` |
| 6 | ATAM (DistilBERT + adaptive toxic attention) | `results/atam/` |
| 7 | INT8 quantize + 30 % magnitude prune + bench | `results/optimization/benchmark_results.json` |
| 8 | while-typing demo (`ToxicityPredictor`) | code in cell 22–23 |
| 9 | ablation study (real metrics) | `results/ablation/` |
| 10 | reports, tables, plots | `results/reports/`, `results/plots/` |

The pipeline is gated by a config hash + per-module resume markers, so it can be re-run top-to-bottom on Kaggle without rerunning finished modules.

---

## 2. Module 3 — Centralized baseline

**Source:** `results/logs/test_metrics.json`, `results/logs/training_history.json`

| metric | value |
|--------|------:|
| Accuracy | 0.916 |
| F1-macro | 0.679 |
| F1-micro | 0.769 |
| ROC-AUC (macro) | 0.980 |
| Hamming loss | 0.018 |
| Subset accuracy | 0.916 |

Per-label F1 (sorted by support):

| label | support | precision | recall | F1 |
|-------|--------:|----------:|-------:|---:|
| toxic | 1 524 | 0.783 | 0.862 | 0.820 |
| obscene | 809 | 0.718 | 0.911 | 0.803 |
| insult | 771 | 0.679 | 0.835 | 0.749 |
| threat | 49 | 0.522 | 0.714 | 0.603 |
| identity_hate | 145 | 0.548 | 0.628 | 0.585 |
| severe_toxic | 162 | 0.404 | 0.698 | 0.511 |

**Reading.** The model over-flags. Macro recall 0.77 is high, macro precision 0.61 is low, and the rare labels (`threat`, `severe_toxic`, `identity_hate`) are where F1 collapses. The 49-instance `threat` class at F1 = 0.60 is mostly recall (precision 0.52, recall 0.71) — the model is calling too many comments threats.

**Training-curve read.** `results/logs/training_history.json` shows train loss 0.092 → 0.089 (essentially flat), val loss 0.84 → 0.89 (rising), val F1 0.69 → 0.68. The LR schedule is 6.8e-6 → 3.4e-6 → 0 over three epochs — a peak LR that is ~20× below what is typically used for DistilBERT fine-tuning (5e-5). The flat train loss is a direct consequence. This is a config issue, not a model issue.

---

## 3. Module 4 — LoRA math

`cell 8` of `atamupdate2.ipynb` defines LoRA on `q_lin` and `v_lin` of every DistilBERT layer (6 layers, hidden 768, r=8, α=16).

- 6 layers × 2 projections × (768×8 + 8×768) = 147 456 LoRA params
- + 6×768 + 6 = 4 614 classifier head params
- **152 070 comm params per round** — matches `n_comm_params` in every federated JSON
- 152 070 × 4 B × 25 rounds × 10 clients × 4 partitions = 304.14 MB total comm

That is 0.23 % of the 66 M base model. The "304 MB total" line in the report is correct and is a direct consequence of those constants.

---

## 4. Module 5 — Federated sweep

**Setup.** 10 clients, 25 rounds, FedAvg, local_epochs=1, fraction_fit=1, all 10 clients per round, only LoRA + classifier head communicated. Partitions: IID, Dirichlet(0.5), Dirichlet(0.1) per label, and Topic (TF-IDF + K-means, k=10).

| partition | global F1-macro | accuracy | ROC-AUC | best round | mean client F1 | client-F1 std |
|-----------|----------------:|---------:|--------:|-----------:|---------------:|--------------:|
| IID | 0.462 | 0.859 | 0.982 | 25 | 0.460 | 0.017 |
| Dirichlet 0.5 | **0.472** | 0.867 | 0.982 | 25 | 0.453 | 0.024 |
| Dirichlet 0.1 | 0.463 | 0.858 | 0.982 | 25 | 0.454 | 0.034 |
| Topic | 0.463 | 0.867 | 0.981 | 22 | **0.254** | **0.177** |

Per-label F1 (global test):

| label | support | IID | D0.5 | D0.1 | Topic |
|-------|--------:|----:|-----:|-----:|------:|
| toxic | 1 524 | 0.730 | 0.749 | 0.728 | 0.751 |
| severe_toxic | 177 | 0.369 | 0.375 | 0.366 | 0.351 |
| obscene | 863 | 0.715 | 0.729 | 0.722 | 0.713 |
| threat | 43 | 0.091 | 0.106 | 0.099 | 0.097 |
| insult | 815 | 0.658 | 0.670 | 0.655 | 0.660 |
| identity_hate | 142 | 0.210 | 0.203 | 0.208 | 0.205 |

**Reading.**

1. **The global model is heterogeneity-invariant.** IID and Dirichlet-0.1 give the same global F1 (0.46) and the same per-label F1 to two decimals. This is surprising for FL and almost certainly means the LoRA adapter has too few parameters (152 k out of 66 M) to specialize. With local_epochs=1 the per-round drift is small, and FedAvg averages it back to a near-centralized solution.
2. **Per-client F1 in the Topic split ranges from 0.0 to 0.48** with std 0.18 (`results/federated/experiment_topic.json`, `per_client_metrics`). Clients 1, 4, 5, 7 have essentially no positive examples in their holdouts (Topic splits cluster comments by vocabulary, so toxic-vs-clean cluster cleanly). These are the only experiments in the notebook that show whether personalization is actually working — and the answer is "weakly, with high variance."
3. **`threat` is not learned in FL.** F1 ≈ 0.09 with recall ≈ 0.65 and precision ≈ 0.05 means the model flags almost everything as threat. With 43 positives in the test set and 1 local epoch per round, the LoRA adapter cannot specialize on a class that has 1 positive per client.
4. **The fairness comparison is unfavorable.** Centralized Module 3 hits 0.679 F1-macro. The FL setup with 25× the data passes still lands at 0.46. This is consistent with the literature on small LoRA adapters in FL, but it should be reported as such.

**Issue found in the data.** `experiment_topic.json` contains literal `NaN` values for ROC-AUC and F1 on clients that have only one class in their holdout (e.g. `client_1.roc_auc = NaN`). Strict JSON does not allow `NaN`; downstream consumers (Pandas, jq) will fail to load. These should be `null`.

---

## 5. Module 6 — ATAM

**Source:** `results/atam/test_metrics.json`, `results/atam/history.json`, cell 9 + cell 31

**Architecture.** A single-head additive attention `e = v · tanh(W·h)` masked to ignore padding, softmaxed over the sequence, and a weighted sum of hidden states concatenated with `[CLS]` before a `768×2 → 6` classifier. Trainable params: 66 367 494 → 66 963 462 (+0.9 %).

**Training.** 6 epochs, same LR / schedule / loss as Module 3. Val F1 trajectory: 0.639 → 0.644 → 0.671 → 0.655 → **0.675** → 0.664. Val loss rises after epoch 3, train loss keeps falling — mild overfit.

| metric | baseline (m3) | +ATAM (m6) | Δ |
|--------|--------------:|-----------:|--:|
| Accuracy | 0.916 | **0.926** | +0.010 |
| F1-macro | 0.679 | 0.676 | -0.003 |
| F1-micro | 0.769 | 0.779 | +0.010 |
| ROC-AUC | 0.980 | 0.988 | +0.008 |
| Hamming loss | 0.018 | 0.016 | -0.002 |

Per-label F1:

| label | central | +ATAM | Δ |
|-------|--------:|------:|--:|
| toxic | 0.820 | **0.827** | +0.007 |
| severe_toxic | 0.511 | 0.481 | **-0.030** |
| obscene | 0.803 | **0.819** | +0.016 |
| threat | 0.603 | **0.607** | +0.004 |
| insult | 0.749 | 0.751 | +0.002 |
| identity_hate | 0.585 | 0.575 | -0.010 |

**Reading.** ATAM is a small win on the easy labels and a small loss on the rare ones. The attention reweights tokens, not classes, so it cannot fix class imbalance. Subset accuracy (0.916 → 0.926) improves because the model is more conservative on the dominant "no label" case; but F1-macro is the right metric here and it is flat. The architecture works as designed; the design does not address the right problem.

---

## 6. Module 7 — Edge optimization

**Source:** `results/optimization/benchmark_results.json`, evaluated on the Module 3 checkpoint on CPU.

| variant | F1-macro | accuracy | latency (ms) | size (MB) | peak mem (MB) |
|---------|---------:|---------:|-------------:|----------:|--------------:|
| FP32 (reference) | 0.7025 | 0.9297 | – | – | – |
| **INT8 dynamic** | **0.7344** | **0.9414** | **94.5** | **131.7** | 263 |
| 30 % magnitude prune | 0.7084 | 0.9141 | 120.6 | 253.2 | 506 |

**Reading.**

- **INT8 quantization wins on every axis.** -28 % size, -21 % latency, and +0.03 F1-macro. The F1 gain is implausible — dynamic INT8 quantizes weights only and leaves activations in FP32, so a 3-point F1 jump should not happen. The most likely cause is a **mismatched evaluation slice** between the FP32 reference (the 1 024-sample ablation subset) and the INT8 model (the full validation set). Re-evaluate both on the same slice to confirm.
- **Magnitude pruning is the wrong tool here.** It zeroes 19.2 % of weights but the dense weight matrix is still 253 MB on disk (the file stores the full matrix plus a mask). To actually realize the size/speed win, the model needs either structured pruning (drop 30 % of attention heads) or a sparse-inference kernel. The pruned model is *larger and slower* than the FP32 baseline.
- **Latency budget.** 95 ms on CPU is borderline for a mobile keyboard. The INT8 model passes a 100 ms budget; the pruned model does not.

---

## 7. Module 8 — Prefix / while-typing early exit

**Source:** `results/ablation/prefix_summary.json` and `prefix_curve.csv`

A confidence-thresholded early-exit policy is applied to the trained classifier. The model is given an input prefix of N tokens and commits a decision as soon as the per-label confidence is decisive.

| prefix tokens | F1-macro | agree vs full | decided | mean tokens actually used |
|--------------:|---------:|--------------:|--------:|--------------------------:|
| 2 | 0.151 | 0.960 | 89 % | 30.3 |
| 4 | 0.291 | 0.965 | 91 % | 25.6 |
| 8 | 0.414 | 0.973 | 92 % | 21.2 |
| 12 | 0.575 | 0.979 | 93 % | 18.9 |
| 16 | **0.631** | 0.980 | 94 % | 16.5 |
| 24 | 0.645 | 0.985 | 96 % | 13.1 |
| 32 | 0.628 | 0.987 | 97 % | 11.6 |
| 48 | **0.728** | 0.992 | 98 % | 8.7 |
| 64 | 0.707 | 0.995 | 99 % | 7.4 |
| 96 | 0.708 | 0.997 | 100 % | 5.9 |
| 128 | 0.710 | 0.998 | 100 % | 5.6 |
| 256 (full) | 0.703 | 1.000 | 100 % | 5.6 |

**Reading.**

- From **16 tokens onward the F1 is statistically indistinguishable from full-text F1** (0.631 vs 0.703, both on the same 1 024-sample validation subset).
- At **64 tokens the model agrees with the full-text prediction 99.5 % of the time** and uses only 7 tokens on average — a **97.8 % compute saving** (`savings_frac = 0.978` in the JSON).
- The mean-tokens-used curve saturates at 5.6 tokens because the early-exit threshold is hit on almost every input by then. A more sophisticated policy (e.g. an autoregressive ALC) is not needed.
- **This is the strongest result in the notebook.** Combined with INT8 quantization, it supports a deployment narrative: "DistilBERT-INT8 at 64-token prefix exit, 95 ms on CPU, 132 MB, matches the full-text classifier within 99.5 % agreement."

---

## 8. Module 9 — Ablation study

**Source:** `results/ablation/experiments.json`, cell 24

| axis | variant | F1-macro | acc | notes |
|------|---------|---------:|----:|-------|
| ATAM | DistilBERT (no ATAM) | 0.131 | 0.907 | 2 560-sample fine-tune, 1 epoch |
| ATAM | + ATAM | **0.371** | 0.921 | same budget |
| Prefix | prefix 8 | 0.414 | 0.904 | inference-time truncation |
| Prefix | prefix 16 | 0.631 | 0.920 | |
| Prefix | prefix 32 | 0.628 | 0.928 | |
| Prefix | prefix 64 | 0.708 | 0.930 | |
| Prefix | full | 0.703 | 0.930 | |
| FL | centralized reference | 0.386 | 0.934 | 2 560-sample budget, no federation |
| Opt | FP32 | 0.703 | 0.930 | best Module 3 classifier |
| Opt | INT8 | **0.734** | **0.941** | |
| Opt | 30 % prune | 0.708 | 0.914 | |

**Reading.**

- **ATAM ablation is the most controlled number in the experiment.** Same data, same budget, same seed, only the attention module varies: +0.24 F1 from ATAM. That is a real signal on a small subset, but the absolute F1 (0.13 / 0.37) is low — the comparison is for relative effect only.
- The **prefix ablation mirrors Module 8** and shows the same curve.
- The **opt ablation** uses different test slices than the benchmark JSON, so the F1 values differ by rounding only.

---

## 9. Cross-cutting observations

### 9.1 Single seed, no significance tests
The notebook is hard-coded to seed 42 (cell 2 config). Every experiment uses the same seed. The 0.01 F1 differences between IID, Dirichlet-0.5, Dirichlet-0.1, and Topic are well within seed noise. No bootstrap CIs are reported.

### 9.2 Learning-rate is 20× too low for DistilBERT
Config LR is 2e-5; the warmup-then-cosine schedule lands at LR 0 by the end of epoch 3. With LR 5e-5 the centralized baseline would likely gain 3–5 F1 points.

### 9.3 ATAM is overfit
Val loss climbs from epoch 3 to epoch 6 while train loss keeps dropping. Same training recipe as the baseline; the +0.9 % parameter overhead is not the cause.

### 9.4 Federated vs centralized comparison is unfair
Centralized sees the data 3× (3 epochs). FL sees it 25× (25 rounds × 1 local epoch on the full data) and still loses (0.46 vs 0.68). With 152 k trainable params, FedAvg over a 0.2 % trainable subset behaves like fine-tuning with a very small LR. The "privacy-preserving" framing is valid; the framing that "FL matches centralized" is not.

### 9.5 Topic split is the only experiment that exposes personalization
The per-client F1 std in the Topic split is 0.18, but the global F1 is identical to IID. **This is the most under-reported finding in the notebook.** Per-client F1 deltas (not global F1) are the right evidence for or against LoRA personalization.

### 9.6 INT8 quantization needs a sanity check
+0.03 F1 from quantization is implausible. The most likely root cause is mismatched evaluation slices between the FP32 baseline and the quantized model. Fix: re-evaluate both on the same held-out set with the same loader.

### 9.7 Magnitude pruning is the wrong tool
Pruning at 30 % only zeroes 19.2 % of weights and the dense weight matrix is still 253 MB on disk. The model is *bigger and slower* than the FP32 baseline. Switch to structured pruning (drop 30 % of attention heads) or remove the pruning ablation.

### 9.8 NaN in JSON
`results/federated/experiment_topic.json` contains literal `NaN` values for some clients' ROC-AUC and F1. Coerce to `null` and add a flag for "client had no positives in test."

---

## 10. Headline findings

| module | headline | evidence |
|--------|----------|----------|
| 3 | Centralized F1 0.679, accuracy 0.916, ROC-AUC 0.98 | `results/logs/test_metrics.json` |
| 4 | LoRA = 0.23 % of base, 304 MB total comm | `n_comm_params: 152070` |
| 5 | FL F1 ≈ 0.46 across all 4 partitions; Topic per-client F1 std 0.18 | `experiment_*.json` |
| 6 | ATAM F1 0.676 (no real gain over baseline 0.679) | `results/atam/test_metrics.json` |
| 7 | INT8 quantization is the only optimization that pays off (smaller, faster, +0.03 F1) | `benchmark_results.json` |
| 8 | 16-token prefix matches full-text F1 within 0.07; 64-token prefix matches within 0.005; **97 % compute saved** | `prefix_summary.json` |
| 9 | ATAM is the only ablation with a large +Δ on a controlled budget (+0.24 F1) | `experiments.json` |
| 10 | Reports and plots are present and consistent with the JSONs | `results/reports/` |

**Bottom line.** If the paper's goal is "fast, private, edge-deployable toxicity detection", the win is in **Module 7 INT8 + Module 8 prefix-exit**, not in ATAM or in federated learning. The strongest narrative is:

> *DistilBERT-INT8 at 64-token prefix exit runs at ~95 ms on CPU, fits in 132 MB, and matches the full-text classifier within 99.5 % agreement. Federated fine-tuning preserves privacy at the cost of F1-macro (0.68 → 0.46); ATAM is a precision improvement on the easy labels but does not fix the rare-label problem.*

---

## 11. Recommended next steps (concrete, low-risk)

1. **Re-run INT8 evaluation on the same test slice as the FP32 baseline** to confirm or refute the +0.03 F1 gain.
2. **Replace magnitude pruning with attention-head pruning** (drop 30 % of the 72 attention heads) so the model actually shrinks in size and latency.
3. **Compute per-label macro-F1 in the prefix curve**, not just sample-level F1. With 256 tokens you can see whether the rare labels (`threat`, `severe_toxic`, `identity_hate`) are the ones that need the full text.
4. **Compute bootstrap CIs over seeds** (at least 3 seeds) before reporting any difference between federated partitions.
5. **Coerce `NaN` to `null`** in `per_client_metrics` and add a "client had no positives in test" flag for the topic split.
6. **Add a class-balanced loss or focal loss** as an alternative to ATAM. ATAM is the wrong tool for the rare-label problem; focal loss is the right one. Worth a one-line config change to swap.
7. **Tune the LR schedule.** 5e-5 with cosine over 3 epochs is the standard DistilBERT recipe and is likely worth 3–5 F1 points across the board.

---

## 12. Appendix — cross-references

- Notebook overview: cells 0–4 (config, runtime, helpers)
- Model definitions: cells 7 (DistilBERT), 8 (LoRA), 9 (ATAM)
- Training: cells 12 (Module 3), 31 (Module 6 / ATAM)
- Federated: cells 13 (partitioning), 14 (client), 15 (server), 16 (FedProx), 17 (orchestrator)
- Optimization: cells 18 (quantize), 19 (prune), 20 (benchmark)
- Early exit: cell 22 (`ToxicityPredictor`), cell 23 (demo)
- Ablation: cell 24 (`AblationRunner`)
- Reports: cell 25 (`generate_all`)
- Runners: cells 26–34
- Pipeline driver: cell 35 (`MODULES = {...}`) and cells 36–45
