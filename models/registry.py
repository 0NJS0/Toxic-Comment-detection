"""
Model Registry — Unified Factory for All Toxicity Detection Models
====================================================================

Provides a single `create_model(config)` function that:
  1. Selects the correct model class (DistilBERT, TinyBERT, MobileBERT)
     based on the model name in config.
  2. Optionally injects LoRA adapters if personalization.method == "lora".
  3. Freezes the base model so only LoRA parameters are trainable.
"""

import torch.nn as nn
from typing import Optional

from models.distilbert import DistilBERTForMultiLabelClassification
from models.tinybert import TinyBERTForMultiLabelClassification
from models.mobilebert import MobileBERTForMultiLabelClassification
from models.lora import inject_lora, freeze_base_model, count_lora_params

MODEL_REGISTRY = {
    "distilbert": DistilBERTForMultiLabelClassification,
    "tinybert": TinyBERTForMultiLabelClassification,
    "mobilebert": MobileBERTForMultiLabelClassification,
}


def resolve_model_key(model_name: str) -> str:
    """Map a HuggingFace model name to our registry key."""
    name_lower = model_name.lower()
    if "tinybert" in name_lower:
        return "tinybert"
    elif "mobilebert" in name_lower or "mobile" in name_lower:
        return "mobilebert"
    else:
        return "distilbert"


def create_model(
    config: dict,
    model_name: Optional[str] = None,
    num_labels: Optional[int] = None,
    dropout: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Create a toxicity classification model from configuration.

    Parameters
    ----------
    config : dict
        Full project configuration (reads model, personalization sections).
    model_name : str, optional
        Override config["model"]["name"].
    num_labels : int, optional
        Override config["model"]["num_labels"].
    dropout : float, optional
        Override config["model"]["dropout"].
    cache_dir : str, optional
        Override config["model"]["cache_dir"].

    Returns
    -------
    nn.Module
        The initialized model (with LoRA injected if configured).
    """
    model_cfg = config["model"]
    model_name = model_name or model_cfg["name"]
    num_labels = num_labels or model_cfg["num_labels"]
    dropout = dropout or model_cfg.get("dropout", 0.1)
    cache_dir = cache_dir or model_cfg.get("cache_dir")

    key = resolve_model_key(model_name)
    ModelClass = MODEL_REGISTRY[key]

    model = ModelClass(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        cache_dir=cache_dir,
    )

    # Inject LoRA if configured
    personalize = config.get("personalization", {})
    if personalize.get("method") == "lora":
        # Freeze base encoder FIRST so LoRA params (added next) remain trainable
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


def create_model_variant(
    config: dict,
    variant: str,
    lora: bool = False,
) -> nn.Module:
    """
    Create a specific model variant with overrides.

    Parameters
    ----------
    config : dict
        Base config.
    variant : str
        One of "distilbert", "tinybert", "mobilebert".
    lora : bool
        Whether to inject LoRA.

    Returns
    -------
    nn.Module
    """
    zoo = config["model"].get("zoo", {})
    model_name = zoo.get(variant, f"{variant}-base-uncased")

    # Temporarily override personalization
    orig_personalize = config.get("personalization", {})
    if lora:
        config["personalization"] = config.get("personalization", {})
        config["personalization"]["method"] = "lora"
    else:
        config["personalization"] = {"method": "none"}

    model = create_model(config, model_name=model_name)

    config["personalization"] = orig_personalize
    return model
