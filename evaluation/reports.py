import json
import numpy as np
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_training_curves(output_dir: str = "results/reports"):
    import matplotlib.pyplot as plt
    metrics_path = Path("results/logs/final_metrics.json")
    if not metrics_path.exists():
        print("No training metrics found. Skip.")
        return
    metrics = load_json(metrics_path)
    epochs = list(range(1, len(metrics.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if "train_loss" in metrics and epochs:
        axes[0].plot(epochs, metrics["train_loss"], marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
    if "val_f1_macro" in metrics and epochs:
        axes[1].plot(epochs, metrics["val_f1_macro"], marker="o", color="green")
        axes[1].set_title("Validation F1 (macro)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1-macro")
    if "val_roc_auc" in metrics and epochs:
        axes[2].plot(epochs, metrics["val_roc_auc"], marker="o", color="orange")
        axes[2].set_title("Validation ROC-AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUROC")
    plt.tight_layout()
    out = Path(output_dir) / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_ablation_results(output_dir: str = "results/reports"):
    import matplotlib.pyplot as plt
    ablation_path = Path("results/ablation/experiments.json")
    if not ablation_path.exists():
        print("No ablation results found. Skip.")
        return
    data = load_json(str(ablation_path))
    results = data.get("results", [])
    axes_map = {}
    for r in results:
        axis = r["axis"]
        if axis not in axes_map:
            axes_map[axis] = []
        axes_map[axis].append(r)
    for axis, exps in axes_map.items():
        variants = [e["variant"] for e in exps]
        f1s = [e["f1_macro"] for e in exps]
        fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.5), 4))
        colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))
        bars = ax.bar(range(len(variants)), f1s, color=colors)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("F1-macro")
        ax.set_title(f"Ablation: {axis}")
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        safe_fn = axis.replace(" ", "_").replace("/", "_")
        out = Path(output_dir) / f"ablation_{safe_fn}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


def generate_ablation_table(output_dir: str = "results/reports") -> str:
    ablation_path = Path("results/ablation/experiments.json")
    if not ablation_path.exists():
        return "*(ablation results not available)*"
    data = load_json(str(ablation_path))
    results = data.get("results", [])
    lines = [
        "| Experiment | Axis | Variant | F1-macro | Accuracy |",
        "|-----------|------|---------|----------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['experiment_id']} | {r['axis']} | {r['variant']} | "
            f"{r['f1_macro']:.4f} | {r['accuracy']:.4f} |"
        )
    table = "\n".join(lines)
    out = Path(output_dir) / "ablation_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(table + "\n")
    print(f"Saved: {out}")
    return table


def generate_report(output_dir: str = "results/reports"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# ATAM-FET: Final Report\n")
    lines.append("**Adaptive Toxic Attention Module for Federated Edge-Based Toxicity Detection**\n")
    lines.append("---\n")

    lines.append("## 1. Centralized Training (Module 3)\n")
    metrics_path = Path("results/logs/final_metrics.json")
    if metrics_path.exists():
        m = load_json(str(metrics_path))
        lines.append(f"- **F1-macro:** {m.get('f1_macro', 'N/A')}")
        lines.append(f"- **Accuracy:** {m.get('accuracy', 'N/A')}")
        lines.append(f"- **ROC-AUC:** {m.get('roc_auc', 'N/A')}")
    else:
        lines.append("*(not available)*")
    lines.append("")

    lines.append("## 2. ATAM Module (Module 6)\n")
    atam_path = Path("results/atam/history.json")
    if atam_path.exists():
        h = load_json(str(atam_path))
        best_f1 = max(h.get("val_f1_macro", [0]))
        lines.append(f"- **Best val F1-macro:** {best_f1:.4f}")
        lines.append("- **Architecture:** DistilBERT + ATAM (toxic attention over all tokens)")
        lines.append("- **Key idea:** Learnable attention weights over token hidden states")
    else:
        lines.append("*(not available)*")
    lines.append("")

    lines.append("## 3. Optimization (Module 7)\n")
    bench_path = Path("results/optimization/benchmark_results.json")
    if bench_path.exists():
        bench = load_json(str(bench_path))
        lines.append("| Model | Latency (ms) | Size (MB) | F1 |")
        lines.append("|-------|-------------|----------|-----|")
        for name, data in bench.items():
            if "metrics" in data:
                f1 = data["metrics"].get("f1_macro", "-")
                lat = data.get("latency_ms", "-")
                size = data.get("model_size_mb", "-")
                f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
                lat_str = f"{lat:.2f}" if isinstance(lat, float) else str(lat)
                size_str = f"{size:.2f}" if isinstance(size, float) else str(size)
                lines.append(f"| {name} | {lat_str} | {size_str} | {f1_str} |")
    else:
        lines.append("*(not available)*")
    lines.append("")

    lines.append("## 4. Ablation Study (Module 9)\n")
    lines.append(generate_ablation_table(str(out_dir)))
    lines.append("")

    lines.append("## 5. Summary\n")
    lines.append("1. **ATAM** improves rare-label detection via learnable token weighting")
    lines.append("2. **Prefix-based detection** enables early toxicity warning")
    lines.append("3. **Federated learning** preserves privacy with LoRA personalization")
    lines.append("4. **Edge optimization** (INT8 quantization, pruning) enables on-device deployment")
    lines.append("")

    report = "\n".join(lines)
    out = out_dir / "final_report.md"
    with open(out, "w") as f:
        f.write(report)
    print(f"Saved: {out}")


def generate_all(output_dir: str = "results/reports"):
    print("\nGenerating training curves...")
    plot_training_curves(output_dir)
    print("\nGenerating ablation plots...")
    plot_ablation_results(output_dir)
    print("\nGenerating ablation table...")
    generate_ablation_table(output_dir)
    print("\nGenerating final report...")
    generate_report(output_dir)
    print(f"\nAll reports saved to {output_dir}/")
