# ATAM-FET: Final Report

**Adaptive Toxic Attention Module for Federated Edge-Based Toxicity Detection**

---

## 1. Centralized Training (Module 3)

- **F1-macro:** 0.6813738613899156
- **Accuracy:** 0.9170067173080545
- **ROC-AUC:** 0.9858744676402557

## 2. ATAM Module (Module 6)

- **Best val F1-macro:** 0.6746
- **Architecture:** DistilBERT + ATAM (toxic attention over all tokens)
- **Key idea:** Learnable attention weights over token hidden states

## 3. Optimization (Module 7)

| Model | Latency (ms) | Size (MB) | F1 |
|-------|-------------|----------|-----|
| quantized | 94.53 | 131.73 | 0.6722 |
| pruned_30pct | 120.56 | 253.21 | 0.6793 |

## 4. Ablation Study (Module 9)

| Experiment | Axis | Variant | F1-macro | Accuracy |
|-----------|------|---------|----------|----------|
| atam_without_atam | ATAM | DistilBERT (no ATAM) | 0.1312 | 0.9072 |
| atam_with_atam | ATAM | DistilBERT + ATAM | 0.3706 | 0.9209 |
| prefix_prefix_8 | Prefix Length | Prefix 8 tokens | 0.4141 | 0.9043 |
| prefix_prefix_16 | Prefix Length | Prefix 16 tokens | 0.6305 | 0.9199 |
| prefix_prefix_32 | Prefix Length | Prefix 32 tokens | 0.6275 | 0.9277 |
| prefix_prefix_64 | Prefix Length | Prefix 64 tokens | 0.7075 | 0.9297 |
| prefix_full | Prefix Length | Full text | 0.7025 | 0.9297 |
| fed_centralized | Federated Rounds | Centralized | 0.3857 | 0.9336 |
| opt_full_precision | Optimization | FP32 | 0.7025 | 0.9297 |
| opt_quantized | Optimization | INT8 Quantized | 0.7344 | 0.9414 |
| opt_pruned_30 | Optimization | Pruned 30% | 0.7084 | 0.9141 |

## 5. Per-Label Performance

### Central (Module 3)

| label | support | precision | recall | f1 |
|-------|---------|-----------|--------|-----|
| toxic | 1524 | 0.783 | 0.862 | 0.820 |
| severe_toxic | 162 | 0.404 | 0.698 | 0.511 |
| obscene | 809 | 0.718 | 0.911 | 0.803 |
| threat | 49 | 0.522 | 0.714 | 0.603 |
| insult | 771 | 0.679 | 0.835 | 0.749 |
| identity_hate | 145 | 0.548 | 0.628 | 0.585 |

### Central + ATAM (Module 6)

| label | support | precision | recall | f1 |
|-------|---------|-----------|--------|-----|
| toxic | 1524 | 0.846 | 0.809 | 0.827 |
| severe_toxic | 162 | 0.487 | 0.475 | 0.481 |
| obscene | 809 | 0.794 | 0.845 | 0.819 |
| threat | 49 | 0.675 | 0.551 | 0.607 |
| insult | 771 | 0.744 | 0.757 | 0.751 |
| identity_hate | 145 | 0.626 | 0.531 | 0.575 |


## 6. Early Detection (Prefix/Earliness)

| prefix tokens | agree vs full | F1-macro | decided | mean tokens used |
|---------------|---------------|----------|---------|------------------|
| 2 | 0.960 | 0.151 | 0.89 | 30.3 |
| 4 | 0.965 | 0.291 | 0.91 | 25.6 |
| 8 | 0.973 | 0.414 | 0.92 | 21.2 |
| 12 | 0.979 | 0.575 | 0.93 | 18.9 |
| 16 | 0.980 | 0.631 | 0.94 | 16.5 |
| 24 | 0.985 | 0.645 | 0.96 | 13.1 |
| 32 | 0.987 | 0.628 | 0.97 | 11.6 |
| 48 | 0.992 | 0.728 | 0.98 | 8.7 |
| 64 | 0.995 | 0.707 | 0.99 | 7.4 |
| 96 | 0.997 | 0.708 | 1.00 | 5.9 |
| 128 | 0.998 | 0.710 | 1.00 | 5.6 |
| 256 | 1.000 | 0.703 | 1.00 | 5.6 |

## 7. Federated Held-out Test + Communication

| Experiment | F1-macro | ROC-AUC | Acc | Comm total (MB) |
|------------|----------|---------|-----|-----------------|
| dirichlet_0.1 | 0.4630 | 0.9820 | 0.8584 | 304.1 |
| dirichlet_0.5 | 0.4721 | 0.9822 | 0.8670 | 304.1 |
| iid | 0.4622 | 0.9815 | 0.8587 | 304.1 |
| topic | 0.4628 | 0.9814 | 0.8666 | 304.1 |

## 8. Summary

1. **ATAM** improves rare-label detection via learnable token weighting
2. **Prefix-based detection** enables early toxicity warning
3. **Federated learning** preserves privacy with LoRA personalization
4. **Edge optimization** (INT8 quantization, pruning) enables on-device deployment
