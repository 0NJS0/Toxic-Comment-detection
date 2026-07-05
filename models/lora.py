"""
LoRA: Low-Rank Adaptation for Parameter-Efficient Personalization
==================================================================

LoRA injects trainable low-rank matrices into attention projections.
The base model stays frozen — only LoRA weights are updated during
federated learning. This means:
  - ~24K trainable params vs 66M base model (0.036%)
  - 99.96% less communication in FL
  - Each client learns a tiny personalization delta

Architecture:
    Original: y = Wx
    LoRA:     y = Wx + s * BAx    (A: R^{r×d}, B: R^{d×r})
    Only BA is trained; W is frozen.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class LoRALayer(nn.Module):
    """
    Low-rank decomposition: BA where A∈R^{r×d_in}, B∈R^{d_out×r}.

    The forward pass computes the LoRA bypass and scales it:
        output = (x @ A.T) @ B.T * (alpha / r)
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T) @ self.B.T * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Wraps an existing nn.Linear with a LoRA bypass.

    The original linear layer is frozen (requires_grad=False).
    Only the LoRALayer is trainable.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.lora = LoRALayer(linear.in_features, linear.out_features, r, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def _get_parent(model: nn.Module, target_name: str) -> Tuple[nn.Module, str]:
    """
    Navigate the module hierarchy to find the parent of a named module.

    Example:
        model.distilbert.transformer.layer[0].attention.q_lin
        -> parent = model.distilbert.transformer.layer[0].attention
        -> child_name = "q_lin"
    """
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        elif part.startswith("[") and part.endswith("]"):
            parent = parent[int(part[1:-1])]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    modules: List[str] = None,
    r: int = 8,
    alpha: float = 16.0,
    verbose: bool = True,
) -> int:
    """
    Inject LoRA adapters into all matching nn.Linear modules in the model.

    Parameters
    ----------
    model : nn.Module
        The model to inject LoRA into (e.g., DistilBERTForMultiLabelClassification).
    modules : list of str, optional
        Substrings to match against module names (e.g., ["q_lin", "v_lin"]).
        Defaults to ["q_lin", "v_lin"].
    r : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    verbose : bool
        Print injection log.

    Returns
    -------
    int
        Number of LoRA adapters injected.
    """
    if modules is None:
        modules = ["q_lin", "v_lin", "query", "value"]

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in modules:
                if target in name:
                    parent, child_name = _get_parent(model, name)
                    setattr(parent, child_name, LinearWithLoRA(module, r, alpha))
                    count += 1
                    if verbose:
                        print(f"  LoRA injected: {name} (r={r}, alpha={alpha})")
                    break
    return count


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all trainable LoRA parameters (A and B matrices)."""
    return [p for n, p in model.named_parameters() if "lora" in n]


def get_lora_names(model: nn.Module) -> List[str]:
    """Return names of all LoRA parameters."""
    return [n for n, p in model.named_parameters() if "lora" in n]


def get_lora_weights(model: nn.Module) -> List[torch.Tensor]:
    """Extract LoRA weights as a flat list of tensors (for FL communication)."""
    return [p.data.clone() for p in get_lora_params(model)]


def set_lora_weights(model: nn.Module, weights: List[torch.Tensor]) -> None:
    """Set LoRA weights from a flat list of tensors (received from server)."""
    params = get_lora_params(model)
    for param, weight in zip(params, weights):
        param.data.copy_(weight)


def count_lora_params(model: nn.Module) -> int:
    """Count total LoRA parameters."""
    return sum(p.numel() for p in get_lora_params(model))


def freeze_base_model(model: nn.Module) -> None:
    """Freeze the base transformer encoder only (not the classification head).

    For DistilBERT/TinyBERT/MobileBERT models, the base encoder is stored
    in `model.distilbert`, `model.tinybert`, or `model.mobilebert`.
    The classifier head (`model.classifier`) remains trainable so it can
    be learned during federated training.
    """
    base_attr = None
    for attr in ["distilbert", "tinybert", "mobilebert"]:
        if hasattr(model, attr):
            base_attr = attr
            break

    if base_attr is None:
        print("  Warning: could not find base encoder attribute to freeze")
        return

    base = getattr(model, base_attr)
    for param in base.parameters():
        param.requires_grad_(False)
