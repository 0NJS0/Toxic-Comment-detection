import torch.nn as nn
from typing import Optional

from models.distilbert import DistilBERTForMultiLabelClassification
from models.atam import DistilBERTWithATAM
from models.lora import inject_lora, freeze_base_model, count_lora_params


def create_model(
    config: dict,
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    dropout: Optional[float] = None,
    cache_dir: Optional[str] = None,
    use_atam: Optional[bool] = None,
) -> nn.Module:
    model_cfg = config["model"]
    model_name = model_name or model_cfg["name"]
    num_labels = num_labels or model_cfg["num_labels"]
    dropout = dropout or model_cfg.get("dropout", 0.1)
    cache_dir = cache_dir or model_cfg.get("cache_dir")
    use_atam = use_atam if use_atam is not None else model_cfg.get("use_atam", False)

    if use_atam:
        from models.atam import DistilBERTWithATAM
        model = DistilBERTWithATAM(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            cache_dir=cache_dir,
        )
    else:
        model = DistilBERTForMultiLabelClassification(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            cache_dir=cache_dir,
        )

    personalize = config.get("personalization", {})
    if personalize.get("method") == "lora":
        freeze_base_model(model)
        lora_cfg = personalize
        num_injected = inject_lora(
            model,
            modules=lora_cfg.get("lora_modules", ["q_lin", "v_lin"]),
            r=lora_cfg.get("lora_r", 8),
            alpha=lora_cfg.get("lora_alpha", 16),
            verbose=True,
        )
        lora_count = count_lora_params(model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  LoRA adapters injected: {num_injected}")
        print(f"  Trainable (LoRA + head): {trainable:,} params")
        print(f"  Frozen (base):           {total - trainable:,} params")
        print(f"  Trainable ratio:         {100 * trainable / total:.4f}%")

    return model
