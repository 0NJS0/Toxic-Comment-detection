import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from models.registry import create_model
from training.train import load_tokenized_dataset, collate_fn
from optimization.quantize import quantize_model_dynamic
from optimization.prune import prune_model_magnitude, compute_pruning_sparsity
from optimization.benchmark import benchmark_model, format_benchmark_table, BenchmarkResult
from evaluation.metrics import compute_metrics


def main():
    print("=" * 60)
    print("MODULE 7: Model Optimization (Quantization, Pruning)")
    print("=" * 60)

    config = load_config("configs/config.yaml")
    set_random_seed(config["project"]["seed"])
    ensure_dir("results/optimization")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = load_tokenized_dataset(config)

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
    )

    ckpt_path = Path("results/checkpoints/checkpoint_epoch_5.pt")
    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model_state = checkpoint["model_state_dict"]
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')}")
    else:
        print("No checkpoint found. Using untrained model (metrics will be random).")
        model_state = None

    if model_state is not None:
        model = create_model(config)
        model.load_state_dict(model_state)
        model.to(device)
    else:
        model = create_model(config).to(device)
    model.eval()

    sample = dataset["validation"][0]
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)
    attn_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)

    all_results = {}

    print("\n[1/2] Quantization...")
    quantized = quantize_model_dynamic(model, dtype=torch.qint8)
    q_bench = benchmark_model(
        quantized, input_ids, attn_mask, device=device, name="Quantized (INT8)",
    )
    all_logits, all_labels = [], []
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
    q_metrics = compute_metrics(all_logits, all_labels)
    q_bench.f1_macro = q_metrics["f1_macro"]
    q_bench.accuracy = q_metrics["accuracy"]
    all_results["quantized"] = q_bench.to_dict()
    all_results["quantized"]["metrics"] = q_metrics
    print(f"  F1: {q_metrics['f1_macro']:.4f} | Latency: {q_bench.latency_ms:.2f}ms | Size: {q_bench.model_size_mb:.2f}MB")

    print("\n[2/2] Pruning...")
    if model_state is not None:
        pruned = create_model(config)
        pruned.load_state_dict(model_state)
    else:
        pruned = create_model(config)
    prune_model_magnitude(pruned, amount=0.3, layers_to_prune="linear", make_permanent=True)
    sparsity = compute_pruning_sparsity(pruned)
    print(f"  Sparsity: {sparsity['sparsity']:.2%} ({sparsity['size_mb']:.2f}MB)")
    p_bench = benchmark_model(
        pruned, input_ids, attn_mask, device=device, name="Pruned (30%)",
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
    p_metrics = compute_metrics(all_logits, all_labels)
    p_bench.f1_macro = p_metrics["f1_macro"]
    p_bench.accuracy = p_metrics["accuracy"]
    all_results["pruned_30pct"] = p_bench.to_dict()
    all_results["pruned_30pct"]["sparsity"] = sparsity
    all_results["pruned_30pct"]["metrics"] = p_metrics
    print(f"  F1: {p_metrics['f1_macro']:.4f} | Latency: {p_bench.latency_ms:.2f}ms | Size: {p_bench.model_size_mb:.2f}MB")

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(format_benchmark_table([q_bench, p_bench]))

    with open("results/optimization/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("MODULE 7 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
