"""
Model Pruning for Mobile Deployment
=====================================

Magnitude-based unstructured pruning.

Removes a fraction of weights with the smallest absolute values.
This reduces model size at the cost of some accuracy.

The pruned model can then be fine-tuned to recover accuracy
(pruning + fine-tuning loop, like Lottery Ticket Hypothesis).
"""

import torch
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Tuple


def prune_model_magnitude(
    model: torch.nn.Module,
    amount: float = 0.3,
    layers_to_prune: str = "linear",
    make_permanent: bool = True,
) -> torch.nn.Module:
    """
    Apply unstructured magnitude pruning to specified layers.

    Parameters
    ----------
    model : nn.Module
        Model to prune (in-place).
    amount : float
        Fraction of weights to prune (0.0 - 1.0).
    layers_to_prune : str
        Which layer types to prune:
          - "linear": nn.Linear layers (default)
          - "attention": attention projection layers
          - "all": all named modules containing 'weight'
    make_permanent : bool
        If True, call prune.remove() to make pruning permanent
        (removes the pruning mask, saves the pruned weights).

    Returns
    -------
    nn.Module: pruned model (same object, modified in-place).
    """
    model.eval()
    pruned_modules = []

    if layers_to_prune == "linear":
        target_modules = [
            module for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        ]
    elif layers_to_prune == "attention":
        target_modules = [
            module for name, module in model.named_modules()
            if any(x in name for x in ["q_lin", "k_lin", "v_lin", "out_lin"])
        ]
    else:
        target_modules = [
            module for _, module in model.named_modules()
            if hasattr(module, "weight")
        ]

    for module in target_modules:
        prune.l1_unstructured(module, name="weight", amount=amount)
        pruned_modules.append(module)

    if make_permanent:
        for module in pruned_modules:
            prune.remove(module, name="weight")

    return model


def compute_pruning_sparsity(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute sparsity statistics for a (possibly pruned) model.

    Returns
    -------
    dict: {
        "total_params": int,
        "zero_params": int,
        "sparsity": float (fraction of zeros),
        "non_zero_params": int,
        "size_mb": float (estimated size of params in MB)
    }
    """
    total = 0
    zeros = 0

    for param in model.parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()

    non_zero = total - zeros
    sparsity = zeros / total if total > 0 else 0.0

    # Estimated size (FP32: 4 bytes per non-zero param)
    size_bytes = non_zero * 4
    size_mb = size_bytes / (1024 * 1024)

    return {
        "total_params": total,
        "zero_params": int(zeros),
        "sparsity": float(sparsity),
        "non_zero_params": non_zero,
        "size_mb": size_mb,
    }
