#!/usr/bin/env python3
"""
===============================================================================
MODULE 5: Federated Learning (FedPref)
===============================================================================

Runs FedPref with configurable partition strategies:
  - IID (baseline)
  - Dirichlet α=0.5 (moderate non-IID)
  - Dirichlet α=0.1 (extreme non-IID)
  - Topic-based (realistic vocabulary clusters)

Example:
    uv run python module_05_federated.py

To run a single experiment:
    uv run python -c "
        from utils.config import load_config
        from federated.train_federated import run_experiment
        c = load_config('configs/config.yaml')
        c['federated']['partition_type'] = 'dirichlet'
        c['federated']['alpha'] = 0.5
        run_experiment(c)
    "
===============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from federated.train_federated import run_all_experiments


def main():
    print("=" * 60)
    print("FedPref: Federated Learning")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])
    ensure_dir("results/federated")

    # Run all partition experiments
    results = run_all_experiments(config)

    print(f"\n{'='*60}")
    print("MODULE 5 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
