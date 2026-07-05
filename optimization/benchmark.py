"""
Model Benchmarking for FedPref
===============================

Measures latency, memory usage, and model size for a given model
on a given device. This tells us whether the model is suitable
for mobile/keyboard deployment.

Key metrics:
  - **Latency** (ms): time per forward pass
  - **Peak memory** (MB): max memory during inference
  - **Model size** (MB): on-disk size of serialized model
  - **Parameter count**: total and trainable
"""

import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model variant."""
    name: str
    latency_ms: float
    latency_std_ms: float
    peak_memory_mb: float
    model_size_mb: float
    total_params: int
    trainable_params: int
    f1_macro: Optional[float] = None
    accuracy: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "latency_ms": self.latency_ms,
            "latency_std_ms": self.latency_std_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "model_size_mb": self.model_size_mb,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "f1_macro": self.f1_macro,
            "accuracy": self.accuracy,
        }


def format_benchmark_table(results: List[BenchmarkResult]) -> str:
    """Format a list of BenchmarkResults as a markdown table."""
    header = (
        "| Model | Latency (ms) | Peak Mem (MB) | Size (MB) | Params (M) | F1 | Acc |\n"
        "|-------|-------------|--------------|----------|-----------|-----|------|"
    )
    rows = []
    for r in results:
        latency = f"{r.latency_ms:.2f} ± {r.latency_std_ms:.2f}"
        mem = f"{r.peak_memory_mb:.1f}"
        size = f"{r.model_size_mb:.2f}"
        params = f"{r.total_params / 1e6:.2f}"
        f1 = f"{r.f1_macro:.3f}" if r.f1_macro else "-"
        acc = f"{r.accuracy:.3f}" if r.accuracy else "-"
        rows.append(f"| {r.name} | {latency} | {mem} | {size} | {params} | {f1} | {acc} |")

    return header + "\n" + "\n".join(rows)


def get_model_size(model: torch.nn.Module) -> float:
    """
    Estimate model size on disk (MB) by serializing state_dict.
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return size_mb


def benchmark_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: torch.device = torch.device("cpu"),
    name: str = "model",
) -> BenchmarkResult:
    """
    Benchmark a model's latency, memory, and size.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark.
    input_ids : torch.Tensor
        Input tensor (batch_size, seq_len).
    attention_mask : torch.Tensor or None
        Attention mask.
    num_warmup : int
        Number of warmup runs (excluded from timing).
    num_runs : int
        Number of timed runs.
    device : torch.device
        Device to benchmark on.
    name : str
        Name for this benchmark result.

    Returns
    -------
    BenchmarkResult
    """
    model = model.to(device)
    model.eval()

    input_ids = input_ids.to(device)
    mask = attention_mask.to(device) if attention_mask is not None else None

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if mask is not None:
                model(input_ids, attention_mask=mask)
            else:
                model(input_ids)

    # Timed runs
    torch.cuda.empty_cache() if device.type == "cuda" else None

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            if mask is not None:
                model(input_ids, attention_mask=mask)
            else:
                model(input_ids)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    # Peak memory (rough estimate for CPU via model size)
    peak_mem = get_model_size(model) * 2  # approx: params + activations

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = get_model_size(model)

    return BenchmarkResult(
        name=name,
        latency_ms=float(np.mean(latencies)),
        latency_std_ms=float(np.std(latencies)),
        peak_memory_mb=peak_mem / (1024 * 1024) if device.type == "cuda" else peak_mem,
        model_size_mb=model_size,
        total_params=total_params,
        trainable_params=trainable_params,
    )
