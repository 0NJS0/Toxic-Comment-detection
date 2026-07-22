"""
Model Quantization for Mobile Deployment
==========================================

Three strategies, ordered by deployment complexity:

1. **Dynamic Quantization** — weights quantized offline, activations
   quantized on-the-fly at inference. Simplest, no calibration data needed.
   Compatible with CPU-only deployment.

2. **Static Quantization** — both weights and activations quantized.
   Requires calibration data to compute activation ranges.
   Better latency but needs representative data.

3. **Quantization-Aware Training (QAT)** — simulates quantization effects
   during training so the model learns to compensate.
   Best accuracy but requires retraining.

All methods reduce model size ~4x (FP32 -> INT8).
"""

import torch
import torch.ao.quantization as quant
from pathlib import Path
from typing import Optional, Dict, Any


class QuantizedModelWrapper(torch.nn.Module):
    """
    Wraps a quantized model to provide the same interface as the
    original (forward, logits attribute, etc.).
    """
    def __init__(self, quantized_model):
        super().__init__()
        self.model = quantized_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs


def quantize_model_dynamic(
    model: torch.nn.Module,
    dtype: torch.dtype = torch.qint8,
    example_inputs: Optional[tuple] = None,
) -> QuantizedModelWrapper:
    """
    Apply dynamic quantization (weights quantized, activations dynamic).

    Works best for transformer models. Reduces size ~4x with minimal
    accuracy loss on CPU.

    Parameters
    ----------
    model : nn.Module
        The trained model to quantize.
    dtype : torch.dtype
        Quantization dtype (qint8 default).
    example_inputs : tuple or None
        Unused for dynamic quantization (kept for API consistency).

    Returns
    -------
    QuantizedModelWrapper
    """
    model.eval()
    model.cpu()

    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear: dtype},
        dtype=dtype,
        inplace=False,
    )

    return QuantizedModelWrapper(quantized)


def quantize_model_static(
    model: torch.nn.Module,
    calibration_data: torch.Tensor,
    calibration_attention_mask: Optional[torch.Tensor] = None,
) -> QuantizedModelWrapper:
    """
    Apply static quantization (weights + activations quantized).

    Requires calibration data to determine activation ranges.

    Parameters
    ----------
    model : nn.Module
        The trained model to quantize.
    calibration_data : torch.Tensor
        Representative input tensor for calibration.
    calibration_attention_mask : torch.Tensor or None
        Attention mask for calibration.

    Returns
    -------
    QuantizedModelWrapper
    """
    model.eval()
    model.cpu()

    # Set default quantization config
    model.qconfig = torch.ao.quantization.get_default_qconfig("x86")

    # Fuse (optional, for some architectures)
    torch.ao.quantization.fuse_modules(model, [["distilbert.transformer.layer.0.attention.q_lin", "distilbert.transformer.layer.0.attention.out_lin"]], inplace=True)

    # Prepare
    prepared = torch.ao.quantization.prepare(model, inplace=False)

    # Calibrate
    with torch.no_grad():
        for i in range(min(calibration_data.size(0), 100)):
            if calibration_attention_mask is not None:
                prepared(
                    calibration_data[i].unsqueeze(0),
                    attention_mask=calibration_attention_mask[i].unsqueeze(0),
                )
            else:
                prepared(calibration_data[i].unsqueeze(0))

    # Convert
    quantized = torch.ao.quantization.convert(prepared, inplace=False)

    return QuantizedModelWrapper(quantized)


def save_quantized_model(model: QuantizedModelWrapper, path: str):
    """Save quantized model state dict."""
    torch.save(model.state_dict(), path)


def load_quantized_model(
    model_class: type,
    path: str,
    dtype: torch.dtype = torch.qint8,
    **model_kwargs,
) -> QuantizedModelWrapper:
    """Load quantized model state dict."""
    model = model_class(**model_kwargs)
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear: dtype},
        dtype=dtype,
        inplace=False,
    )
    quantized.load_state_dict(torch.load(path, weights_only=True))
    return QuantizedModelWrapper(quantized)
