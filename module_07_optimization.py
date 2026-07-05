#!/usr/bin/env python3
"""
===============================================================================
MODULE 7: Model Optimization (Quantization, Distillation, Pruning)
===============================================================================

Applies and benchmarks three optimization techniques for mobile deployment:

1. **Quantization** — INT8 dynamic quantization (4x size reduction)
2. **Distillation** — TinyBERT student trained from DistilBERT teacher
3. **Pruning** — Magnitude-based unstructured pruning

Each technique is benchmarked for latency, memory, and accuracy.

Run:
    uv run python module_07_optimization.py
===============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import json
import numpy as np
from evaluation.metrics import compute_metrics
from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from models.registry import create_model
from models.tinybert import TinyBERTForMultiLabelClassification
from models.mobilebert import MobileBERTForMultiLabelClassification
from training.train import load_tokenized_dataset, MultiLabelDataCollator
from optimization.quantize import quantize_model_dynamic
from optimization.distill import DistillationTrainer
from optimization.prune import prune_model_magnitude, compute_pruning_sparsity
from optimization.benchmark import benchmark_model, format_benchmark_table, BenchmarkResult
from torch.utils.data import DataLoader


def main():
    print("=" * 60)
    print("FedPref: Model Optimization (Module 7)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])
    ensure_dir("results/optimization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer, train_dataset, val_dataset = load_tokenized_dataset(
        config, val_split=True, seed=config["project"]["seed"],
    )
    label_cols = config["data"]["labels"]
    num_labels = len(label_cols)

    # Load pre-trained DistilBERT
    teacher = create_model(
        config,
        model_name=config["model"]["name"],
        num_labels=num_labels,
    )
    checkpoint = torch.load(
        "results/training/best_model.pt",
        map_location=device,
        weights_only=True,
    )
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.to(device)
    teacher.eval()
    print(f"Teacher loaded (DistilBERT)")

    # Benchmark samples
    sample = val_dataset[0]
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)
    attn_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)
    # Batch of 1 for latency benchmark
    benchmark_inputs = input_ids.repeat(1, 1)
    benchmark_mask = attn_mask.repeat(1, 1)

    all_results = {}

    # ------------------------------------------------------------------ #
    # 1. Quantization
    # ------------------------------------------------------------------ #
    print("\n[1/3] Quantization...")
    quantized = quantize_model_dynamic(teacher, dtype=torch.qint8)
    q_bench = benchmark_model(
        quantized, benchmark_inputs, benchmark_mask,
        device=device, name="Quantized (INT8)",
    )

    # Evaluate quantized accuracy
    all_logits, all_labels = [], []
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    collator = MultiLabelDataCollator(tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collator)
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = quantized(ids, attention_mask=mask)
            logits = out.logits if hasattr(out, "logits") else out
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    q_metrics = compute_metrics(all_labels, all_logits, label_cols=label_cols)
    q_bench.f1_macro = q_metrics["f1_macro"]
    q_bench.accuracy = q_metrics["accuracy"]

    all_results["quantized"] = q_bench.to_dict()
    all_results["quantized"]["metrics"] = q_metrics
    print(f"  F1: {q_metrics['f1_macro']:.4f} | Latency: {q_bench.latency_ms:.2f}ms | Size: {q_bench.model_size_mb:.2f}MB")

    # ------------------------------------------------------------------ #
    # 2. Distillation: TinyBERT student
    # ------------------------------------------------------------------ #
    print("\n[2/3] Distillation (TinyBERT -> student)...")
    student = TinyBERTForMultiLabelClassification(num_labels=num_labels)

    collator = MultiLabelDataCollator(tokenizer)
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        device=device,
        temperature=4.0,
        alpha=0.5,
    )
    distill_history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        num_epochs=5,
        output_dir="results/optimization/distillation",
    )

    # Load best student
    best_student = TinyBERTForMultiLabelClassification(num_labels=num_labels)
    best_student.load_state_dict(
        torch.load("results/optimization/distillation/best_student.pt", map_location=device, weights_only=True)["model_state_dict"]
    )
    best_student.to(device)

    s_bench = benchmark_model(
        best_student, benchmark_inputs, benchmark_mask,
        device=device, name="TinyBERT (Distilled)",
    )

    # Evaluate
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = best_student(ids, attention_mask=mask)
            logits = out.logits if hasattr(out, "logits") else out
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    s_metrics = compute_metrics(all_labels, all_logits, label_cols=label_cols)
    s_bench.f1_macro = s_metrics["f1_macro"]
    s_bench.accuracy = s_metrics["accuracy"]

    all_results["distilled_tinybert"] = s_bench.to_dict()
    all_results["distilled_tinybert"]["metrics"] = s_metrics
    all_results["distillation_history"] = distill_history
    print(f"  F1: {s_metrics['f1_macro']:.4f} | Latency: {s_bench.latency_ms:.2f}ms | Size: {s_bench.model_size_mb:.2f}MB")

    # ------------------------------------------------------------------ #
    # 3. Pruning
    # ------------------------------------------------------------------ #
    print("\n[3/3] Pruning...")
    pruned = create_model(config, model_name=config["model"]["name"], num_labels=num_labels)
    pruned.load_state_dict(checkpoint["model_state_dict"])
    prune_model_magnitude(pruned, amount=0.3, layers_to_prune="linear", make_permanent=True)
    sparsity = compute_pruning_sparsity(pruned)
    print(f"  Sparsity: {sparsity['sparsity']:.2%} ({sparsity['size_mb']:.2f}MB)")

    p_bench = benchmark_model(
        pruned, benchmark_inputs, benchmark_mask,
        device=device, name="Pruned (30%)",
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = pruned(ids, attention_mask=mask)
            logits = out.logits if hasattr(out, "logits") else out
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    p_metrics = compute_metrics(all_labels, all_logits, label_cols=label_cols)
    p_bench.f1_macro = p_metrics["f1_macro"]
    p_bench.accuracy = p_metrics["accuracy"]

    all_results["pruned_30pct"] = p_bench.to_dict()
    all_results["pruned_30pct"]["sparsity"] = sparsity
    all_results["pruned_30pct"]["metrics"] = p_metrics
    print(f"  F1: {p_metrics['f1_macro']:.4f} | Latency: {p_bench.latency_ms:.2f}ms | Size: {p_bench.model_size_mb:.2f}MB")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    results_list = [
        BenchmarkResult(
            name="DistilBERT (Original)",
            latency_ms=0, latency_std_ms=0, peak_memory_mb=0, model_size_mb=0,
            total_params=0, trainable_params=0,
            f1_macro=checkpoint.get("best_f1", q_metrics["f1_macro"]),
            accuracy=q_metrics["accuracy"],
        ),
        q_bench,
        s_bench,
        p_bench,
    ]

    print(format_benchmark_table(results_list))

    with open("results/optimization/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("MODULE 7 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
