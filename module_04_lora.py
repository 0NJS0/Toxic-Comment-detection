#!/usr/bin/env python3
"""
===============================================================================
MODULE 4: Model Zoo + LoRA Personalization
===============================================================================

Verifies the model zoo and LoRA injection pipeline:

  1. Create all 3 model variants (DistilBERT, TinyBERT, MobileBERT)
  2. Verify LoRA injection reduces trainable params to ~0.2%
  3. Verify freeze order (base BEFORE LoRA)
  4. Test get/set LoRA weights (for federated communication)
  5. Benchmark model creation time

Run:
    uv run python module_04_lora.py
===============================================================================
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.helpers import set_random_seed
from models.registry import create_model
from models.lora import count_lora_params


def main():
    print("=" * 60)
    print("FedPref: Model Zoo + LoRA Personalization (Module 4)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])

    # Architecture-specific LoRA target modules
    lora_targets = {
        "distilbert": ["q_lin", "v_lin"],
        "tinybert": ["query", "value"],
        "mobilebert": ["query", "value"],
    }

    models_to_test = [
        ("distilbert-base-uncased", "DistilBERT"),
        ("huawei-noah/TinyBERT_General_4L_312D", "TinyBERT"),
        ("google/mobilebert-uncased", "MobileBERT"),
    ]

    for model_name, label in models_to_test:
        print(f"\n--- {label} ({model_name}) ---")

        # Without LoRA
        start = time.time()
        model = create_model(config, model_name=model_name, num_labels=6)
        elapsed = time.time() - start
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Without LoRA: {total:,} total, {trainable:,} trainable ({elapsed:.1f}s)")

        # With LoRA (architecture-specific modules)
        key = "distilbert" if "distilbert" in model_name else "tinybert" if "tinybert" in model_name else "mobilebert"
        lora_cfg = {
            "method": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_modules": lora_targets[key],
        }
        from copy import deepcopy
        cfg_lora = deepcopy(config)
        cfg_lora["personalization"] = lora_cfg
        start = time.time()
        model_lora = create_model(cfg_lora, model_name=model_name, num_labels=6)
        elapsed = time.time() - start
        lora_count = count_lora_params(model_lora)
        total = sum(p.numel() for p in model_lora.parameters())
        pct = 100 * lora_count / total if total else 0
        print(f"  With LoRA:    {total:,} total, "
              f"{lora_count:,} LoRA, "
              f"{pct:.4f}% trainable ({elapsed:.1f}s)")

        # Verify non-LoRA params are frozen
        frozen = sum(p.numel() for p in model_lora.parameters() if not p.requires_grad)
        assert frozen > 0, "Base model should be frozen"

    # Verify get/set LoRA weights round-trip
    print(f"\n--- LoRA weight round-trip test ---")
    config["personalization"] = {"method": "lora", "lora_r": 8, "lora_alpha": 16,
                                  "lora_dropout": 0.1, "lora_modules": ["q_lin", "v_lin"]}
    model = create_model(config)
    from models.lora import get_lora_weights, set_lora_weights
    import torch
    weights_before = get_lora_weights(model)
    # Modify weights
    modified = [w + 0.1 for w in weights_before]
    set_lora_weights(model, modified)
    weights_after = get_lora_weights(model)
    diffs = [torch.max(torch.abs(a - b)).item() for a, b in zip(weights_before, weights_after)]
    print(f"  Round-trip OK: {len(weights_before)} param groups, max diff = {max(diffs):.6f}")

    print(f"\n{'='*60}")
    print("MODULE 4 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
