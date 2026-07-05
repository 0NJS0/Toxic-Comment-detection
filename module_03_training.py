#!/usr/bin/env python3
"""
================================================================================
MODULE 3: Centralized DistilBERT Training
================================================================================

WHAT THIS MODULE DOES
----------------------
1. Loads the tokenized dataset from Module 2 (data/processed/tokenized/)
2. Creates PyTorch DataLoaders
3. Initializes DistilBERT for multi-label classification
4. Trains with BCEWithLogitsLoss + AdamW + linear schedule
5. Evaluates on validation set after each epoch
6. Saves model checkpoints, training curves, and metrics

WHY THIS MATTERS
----------------
This is the central baseline model. All later experiments
(federated learning, TinyBERT, quantization, prefix detection)
are compared against this centralized DistilBERT performance.

HOW TO RUN
----------
    uv run python module_03_training.py

OUTPUT
------
    results/checkpoints/
    ├── best_model/           # Best model (by validation F1-macro)
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_epoch_2.pt
    └── checkpoint_epoch_3.pt

    results/logs/
    ├── training_history.json
    └── final_metrics.json

    results/plots/
    └── training_curves.png

INDEPENDENT EXECUTION
---------------------
This module loads the tokenized dataset from disk.
Does NOT need Modules 1 or 2 to be rerun.
================================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from training.train import run_training


def main():
    print("=" * 60)
    print("Federated Edge-Based Early Toxicity Detection")
    print("Module 3: Centralized Training")
    print("=" * 60)
    print()

    # Step 1: Load configuration
    print("[1/3] Loading configuration...")
    config = load_config("configs/config.yaml")
    print(f"  Model:      {config['model']['name']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs:     {config['training']['epochs']}")
    print(f"  LR:         {config['training']['learning_rate']}")
    print()

    # Step 2: Set random seed
    print("[2/3] Setting random seed...")
    set_random_seed(config["project"]["seed"])
    print()

    # Step 3: Run training pipeline
    print("[3/3] Running training pipeline...")
    print()
    run_training(config)


if __name__ == "__main__":
    main()
