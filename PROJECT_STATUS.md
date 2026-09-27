# Project Status: ATAM-FET — Toxic Comment Detection

> **Current State:** Modules 1-3 complete (executed). Modules 4-10 coded (not yet executed).

---

## Module Checklist

| # | Module | Status | Files |
|---|--------|--------|-------|
| 1 | Data Preparation | ✅ Complete | `preprocessing/clean.py`, `module_01_data_preparation.py` |
| 2 | Tokenization | ✅ Complete | `preprocessing/tokenize.py`, `module_02_tokenization.py` |
| 3 | Centralized DistilBERT Training | ✅ Complete | `training/train.py`, `models/distilbert.py`, `module_03_training.py` |
| 4 | LoRA Personalization | ⏳ Coded (not run) | `models/lora.py`, `module_04_lora.py` |
| 5 | Federated Learning | ⏳ Coded (not run) | `federated/`, `module_05_federated.py` |
| 6 | ATAM Module | ⏳ Coded (not run) | `models/atam.py`, `module_06_atam.py` |
| 7 | Model Optimization | ⏳ Coded (not run) | `optimization/`, `module_07_optimization.py` |
| 8 | Simulation UI | ⏳ Coded (not run) | `simulation/`, `module_08_simulation.py` |
| 9 | Ablation Study | ⏳ Coded (not run) | `evaluation/ablation.py`, `module_09_ablation.py` |
| 10 | Reports & Plots | ⏳ Coded (not run) | `evaluation/reports.py`, `module_10_reports.py` |

---

## How to Run

```bash
uv sync
uv run python module_01_data_preparation.py
uv run python module_02_tokenization.py
uv run python module_03_training.py
uv run python module_04_lora.py
uv run python module_05_federated.py
uv run python module_06_atam.py
uv run python module_07_optimization.py
uv run streamlit run simulation/app.py
uv run python module_09_ablation.py
uv run python module_10_reports.py
```
