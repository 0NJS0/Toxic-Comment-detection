#!/usr/bin/env python3
"""
===============================================================================
MODULE 9: Ablation Study
===============================================================================

Systematically compares design choices across 6 axes:
  - Model size, LoRA rank, Prefix curriculum, Fed rounds,
    Non-IID severity, Non-IID type

Run:
    uv run python module_09_ablation.py
===============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from evaluation.ablation import AblationRunner


def main():
    print("=" * 60)
    print("FedPref: Ablation Study (Module 9)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])
    ensure_dir("results/ablation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    runner = AblationRunner(config, device)
    results = runner.run_all()

    runner.save("results/ablation/experiments.json")

    print("\n" + "=" * 60)
    print("Ablation Summary")
    print("=" * 60)
    print(runner.summary_table())

    print(f"\n{'='*60}")
    print("MODULE 9 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
