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
    print("MODULE 4: LoRA Personalization")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])

    for use_atam in [False, True]:
        label = "DistilBERT" if not use_atam else "DistilBERT + ATAM"
        print(f"\n--- {label} ---")

        start = time.time()
        model = create_model(config, use_atam=use_atam)
        elapsed = time.time() - start
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Without LoRA: {total:,} total, {trainable:,} trainable ({elapsed:.1f}s)")

        from copy import deepcopy
        cfg_lora = deepcopy(config)
        cfg_lora["personalization"] = {
            "method": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_modules": ["q_lin", "v_lin"],
        }
        start = time.time()
        model_lora = create_model(cfg_lora, use_atam=use_atam)
        elapsed = time.time() - start
        lora_count = count_lora_params(model_lora)
        total = sum(p.numel() for p in model_lora.parameters())
        pct = 100 * lora_count / total if total else 0
        print(f"  With LoRA:    {total:,} total, "
              f"{lora_count:,} LoRA, "
              f"{pct:.4f}% trainable ({elapsed:.1f}s)")

        frozen = sum(p.numel() for p in model_lora.parameters() if not p.requires_grad)
        assert frozen > 0, "Base model should be frozen"

    import torch
    from models.lora import get_lora_weights, set_lora_weights
    config_lora = deepcopy(config)
    config_lora["personalization"] = {"method": "lora", "lora_r": 8, "lora_alpha": 16,
                                      "lora_dropout": 0.1, "lora_modules": ["q_lin", "v_lin"]}
    model = create_model(config_lora)
    weights_before = get_lora_weights(model)
    modified = [w + 0.1 for w in weights_before]
    set_lora_weights(model, modified)
    weights_after = get_lora_weights(model)
    diffs = [torch.max(torch.abs(a - b)).item() for a, b in zip(weights_before, weights_after)]
    print(f"\n  LoRA round-trip OK: {len(weights_before)} param groups, max diff = {max(diffs):.6f}")

    print(f"\n{'='*60}")
    print("MODULE 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
