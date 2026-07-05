# FedPref: Federated Prefix-Based Early Toxicity Detection

Privacy-preserving early toxicity detection for mobile keyboards using personalized federated learning and LoRA-adapted lightweight transformers.

## Core Contribution

**FedPref** is a personalized federated learning framework for *while-typing* toxicity detection:

1. **Personalized FL** — LoRA adapters capture user-specific slang and language patterns; only adapter weights (0.04% of model) leave the device
2. **Prefix-based early detection** — model predicts toxicity from partial text prefixes, flagging harmful content ~30 tokens earlier than full-text models
3. **Mobile-friendly compression** — TinyBERT (14M params) + INT8 quantization = 10 MB on-device model with 45 ms latency

## Architecture

```
┌──────────┐   ┌──────────┐         ┌──────────┐
│ Client 1 │   │ Client 2 │   ...   │ Client N │
│ LoRA_A   │   │ LoRA_B   │         │ LoRA_N   │
└────┬─────┘   └────┬─────┘         └────┬─────┘
     │ LoRA weights │                    │
     └──────────────┴──────────┬─────────┘
                               ▼
                     ┌─────────────────┐
                     │  Flower Server  │
                     │  FedAvg LoRA    │
                     └─────────────────┘

Training:
  Full comment → Prefix sampler [8, 16, 32, 64, 128]
                      │
              Shared encoder (frozen on-device)
                      │
              ┌───────┴───────┐
              │  LoRA adapters│ ← Per-client, FL-aggregated
              └───────┬───────┘
                      │
              [CLS] → Classification head
                      │
                      ▼
              Toxicity prob at EACH prefix length
```

## Quick Start

```bash
uv sync

# Module 1: Data preparation (cleaning)
uv run python module_01_data_preparation.py

# Module 2: Tokenization
uv run python module_02_tokenization.py

# Module 3: Centralized DistilBERT baseline
uv run python module_03_training.py

# Module 4: Model definitions + LoRA (verification)
uv run python -c "from models.registry import create_model; m = create_model(config); print(m.get_num_parameters())"

# Module 5: Federated learning (FedPref)
uv run python module_05_federated.py

# Module 6: Prefix-based early detection training
uv run python module_06_prefix.py

# Module 7: Model optimization (quantization, distillation, pruning)
uv run python module_07_optimization.py

# Module 8: Interactive simulation
streamlit run simulation/app.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Centralized F1-macro (DistilBERT, full text) | **0.698** |
| Federated F1-macro (FedPref, LoRA r=8) | **0.695** |
| Prefix detection point | **~35 tokens** |
| Characters saved per toxic comment | **~85 chars** |
| Communication savings (LoRA vs full model) | **99.96%** |
| Deployed model size (TinyBERT + INT8) | **10 MB** |
| On-device latency | **45 ms** |

## Non-IID Simulation

Two strategies for heterogeneous client simulation:

- **Label-Dirichlet (α=0.5)** — standard FL benchmark; each client gets skewed label distribution
- **Topic-based** — realistic keyboard scenario; clients grouped by vocabulary cluster (gaming slang, politics, etc.)

Both are configurable via `configs/config.yaml`.

## Project Structure

```
├── configs/config.yaml          ← All hyperparameters
├── data/                        ← Raw + processed + partitions
├── preprocessing/               ← Cleaning (M1) + Tokenization (M2)
├── training/                    ← Centralized (M3) + Prefix (M6)
├── models/                      ← DistilBERT + TinyBERT + MobileBERT + LoRA (M4)
├── federated/                   ← Flower clients, server, strategies (M5)
├── evaluation/                  ← Metrics + prefix eval + ablation (M9)
├── optimization/                ← Quantization, distillation, pruning (M7)
├── simulation/                  ← Streamlit interactive demo (M8)
├── results/                     ← Checkpoints, logs, plots, reports (M10)
├── utils/                       ← Config loader, helpers
├── module_0*_*.py               ← Entry points for each module
└── PROJECT_STATUS.md            ← Current state and progress
```

## Citation

If you use this work in your research, please cite:

```
@software{fedpref2025,
  title = {FedPref: Federated Prefix-Based Early Toxicity Detection},
  author = {Your Name},
  year = {2025},
}
```

## License

Research purpose.
