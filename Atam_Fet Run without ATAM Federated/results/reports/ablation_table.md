# Ablation Study Results

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
