#!/usr/bin/env python3
"""
===============================================================================
MODULE 6: Prefix-Based Early Toxicity Detection
===============================================================================

Trains a model to detect toxicity from *partial* text prefixes using
curriculum learning.

Run:
    uv run python module_06_prefix.py
===============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import json
from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from training.prefix_train import train_prefix_model, evaluate_prefix_model
from training.train import load_tokenized_dataset


def main():
    print("=" * 60)
    print("FedPref: Prefix-Based Early Detection (Module 6)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])
    ensure_dir("results/prefix")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer, train_dataset, val_dataset = load_tokenized_dataset(
        config, val_split=True, seed=config["project"]["seed"],
    )
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Train
    history = train_prefix_model(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        device=device,
        output_dir="results/prefix",
    )
    print(f"Training complete. Best F1: {history.get('best_f1', 'N/A'):.4f}")

    # Evaluate at all prefix lengths
    print("\nEvaluating at multiple prefix lengths...")
    from models.registry import create_model
    model = create_model(
        model_name=config["model"]["name"],
        num_labels=len(config["data"]["labels"]),
    )
    model.load_state_dict(
        torch.load("results/prefix/best_model.pt", map_location=device, weights_only=True)["model_state_dict"]
    )
    model.to(device)
    model.eval()

    results = evaluate_prefix_model(
        model=model,
        dataset=val_dataset,
        tokenizer=tokenizer,
        device=device,
        label_cols=config["data"]["labels"],
    )

    print("\nPrefix Evaluation Results:")
    print("-" * 50)
    for key in sorted(results.keys(), key=lambda k: (k == "full", k)):
        m = results[key]
        f1 = m.get("f1_macro", 0)
        acc = m.get("accuracy", 0)
        au = m.get("roc_auc", 0)
        print(f"  {key:>12}: F1={f1:.4f}  Acc={acc:.4f}  AUROC={au:.4f}")

    with open("results/prefix/prefix_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("MODULE 6 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
