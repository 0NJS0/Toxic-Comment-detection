"""
Optimization module for FedPref: quantization, distillation, pruning, benchmarking.
"""

from .quantize import quantize_model_dynamic, quantize_model_static, QuantizedModelWrapper
from .distill import distill_knowledge, DistillationTrainer
from .prune import prune_model_magnitude, compute_pruning_sparsity
from .benchmark import benchmark_model, BenchmarkResult, format_benchmark_table

__all__ = [
    "quantize_model_dynamic",
    "quantize_model_static",
    "QuantizedModelWrapper",
    "distill_knowledge",
    "DistillationTrainer",
    "prune_model_magnitude",
    "compute_pruning_sparsity",
    "benchmark_model",
    "BenchmarkResult",
    "format_benchmark_table",
]
