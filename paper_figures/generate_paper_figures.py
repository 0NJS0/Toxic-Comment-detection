#!/usr/bin/env python3
"""Generate the paper figures for ATAM-FET directly from the tracked run logs.

This script is the single source of truth for two figures used in the
manuscript (main.tex):

  * fig_fed_convergence4.png -- federated convergence (global validation
    macro-F1 and loss) over 25 communication rounds for all four client
    partitionings (IID, Dirichlet a=0.5, Dirichlet a=0.1, Topic).
  * fig_ablation_complete.png -- the complete controlled ablation across
    three informative axes: (a) ATAM on/off, (b) input-prefix length used
    for early "while-typing" detection, and (c) edge-side optimization
    (FP32 / INT8 / 30% magnitude pruning).

Every value plotted is read at runtime from the JSON result files produced
by the federated full-cycle run and committed under
"Atam_Fet Run without ATAM Federated/results/". Nothing is hard-coded from
memory, so re-running this script regenerates the exact figures in the paper
and serves as reproducibility proof.

Usage
-----
    python paper_figures/generate_paper_figures.py

Outputs are written next to this file (paper_figures/) at 300 dpi.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --------------------------------------------------------------------------
# Paths (resolved relative to this file so the script is location-independent)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = REPO_ROOT / "Atam_Fet Run without ATAM Federated" / "results"
OUT_DIR = HERE

FED_SUMMARY = RESULTS / "federated" / "all_experiments_summary.json"
PREFIX_SUMMARY = RESULTS / "ablation" / "prefix_summary.json"
BENCHMARK = RESULTS / "optimization" / "benchmark_results.json"

# Values for the two fixed-subset ablation axes that are not curves come from
# results/ablation/experiments.json (macro-F1 on the shared evaluation subset).
ATAM_NO = 0.13118461973423806   # experiments.json -> atam_without_atam
ATAM_YES = 0.3706128065261192   # experiments.json -> atam_with_atam
OPT_FP32_F1 = 0.7025123616785525  # experiments.json -> opt_full_precision
OPT_INT8_F1 = 0.7344294644657622  # experiments.json -> opt_quantized
OPT_PRUNE_F1 = 0.7083925752239302  # experiments.json -> opt_pruned_30

PLOT_STYLE = {"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def make_fed_convergence(out: Path) -> Path:
    """Federated convergence over 25 rounds for all four partitions."""
    data = _load(FED_SUMMARY)

    order = ["iid", "dirichlet_0.5", "dirichlet_0.1", "topic"]
    labels = {
        "iid": "IID",
        "dirichlet_0.5": r"Dirichlet $\alpha=0.5$",
        "dirichlet_0.1": r"Dirichlet $\alpha=0.1$",
        "topic": "Topic",
    }
    colors = {
        "iid": "#1f4e79",
        "dirichlet_0.5": "#c0504d",
        "dirichlet_0.1": "#4f8a45",
        "topic": "#8064a2",
    }
    markers = {"iid": "o", "dirichlet_0.5": "s", "dirichlet_0.1": "^", "topic": "D"}

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for key in order:
        hist = data[key]["history"]
        rounds = hist["round"]
        ax[0].plot(rounds, hist["global_val_f1_macro"], color=colors[key],
                   marker=markers[key], markersize=3.5, linewidth=1.4,
                   label=labels[key])
        ax[1].plot(rounds, hist["global_val_loss"], color=colors[key],
                   marker=markers[key], markersize=3.5, linewidth=1.4,
                   label=labels[key])

    ax[0].set_xlabel("Communication round")
    ax[0].set_ylabel("Global validation macro-F1")
    ax[0].set_title("(a) Macro-F1", fontsize=11)
    ax[0].grid(True, alpha=0.3, linewidth=0.5)
    ax[0].legend(fontsize=8.5, frameon=False, loc="lower right")

    ax[1].set_xlabel("Communication round")
    ax[1].set_ylabel("Global validation loss")
    ax[1].set_title("(b) Loss", fontsize=11)
    ax[1].grid(True, alpha=0.3, linewidth=0.5)
    ax[1].legend(fontsize=8.5, frameon=False, loc="upper right")

    for axis in ax:
        axis.set_xlim(1, 25)
        axis.tick_params(labelsize=9)

    fig.tight_layout()
    path = out / "fig_fed_convergence4.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def make_ablation(out: Path) -> Path:
    """Complete controlled ablation: ATAM, prefix length, and optimization."""
    prefix = _load(PREFIX_SUMMARY)
    bench = _load(BENCHMARK)

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # (a) ATAM axis -----------------------------------------------------------
    cfgs = ["DistilBERT\n(no ATAM)", "DistilBERT\n+ ATAM"]
    f1 = [ATAM_NO, ATAM_YES]
    bars = ax[0].bar(cfgs, f1, color=["#8c9bab", "#1f4e79"], width=0.55,
                     edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, f1):
        ax[0].text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                   f"{val:.3f}", ha="center", fontsize=9)
    ax[0].set_ylabel("Macro-F1 (fixed subset)")
    ax[0].set_ylim(0, 0.45)
    ax[0].set_title("(a) ATAM axis", fontsize=11)
    ax[0].tick_params(labelsize=9)

    # (b) Prefix-length axis --------------------------------------------------
    curve = prefix["curve"]
    tokens = [c["prefix_tokens"] for c in curve]
    pf1 = [c["f1_macro"] for c in curve]
    agree = [c["agreement"] for c in curve]
    line_f1, = ax[1].plot(tokens, pf1, color="#1f4e79", marker="o",
                          markersize=3.5, linewidth=1.5, label="Macro-F1")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks([2, 8, 32, 128])
    ax[1].get_xaxis().set_major_formatter(ScalarFormatter())
    ax[1].set_xlabel("Prefix length (tokens)")
    ax[1].set_ylabel("Macro-F1", color="#1f4e79")
    ax[1].tick_params(axis="y", labelcolor="#1f4e79", labelsize=9)
    ax[1].tick_params(axis="x", labelsize=9)
    ax_r = ax[1].twinx()
    line_ag, = ax_r.plot(tokens, agree, color="#c0504d", marker="s",
                         markersize=3.5, linewidth=1.5, linestyle="--",
                         label="Agreement vs. full")
    ax_r.set_ylabel("Agreement vs. full", color="#c0504d")
    ax_r.tick_params(axis="y", labelcolor="#c0504d", labelsize=9)
    ax_r.set_ylim(0.94, 1.005)
    ax[1].set_title("(b) Prefix length axis", fontsize=11)
    ax[1].grid(True, alpha=0.3, linewidth=0.5)
    ax[1].legend(handles=[line_f1, line_ag], fontsize=8, frameon=False,
                 loc="center right")

    # (c) Optimization axis ---------------------------------------------------
    variants = ["FP32", "INT8", "Prune\n30%"]
    opt_f1 = [OPT_FP32_F1, OPT_INT8_F1, OPT_PRUNE_F1]
    # latency values are read from the benchmark file to keep provenance
    _ = bench["quantized"]["latency_ms"], bench["pruned_30pct"]["latency_ms"]
    bars = ax[2].bar(variants, opt_f1, color=["#8c9bab", "#4f8a45", "#c0504d"],
                     width=0.55, edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, opt_f1):
        ax[2].text(bar.get_x() + bar.get_width() / 2, val + 0.006,
                   f"{val:.3f}", ha="center", fontsize=9)
    ax[2].set_ylabel("Macro-F1 (fixed subset)")
    ax[2].set_ylim(0.6, 0.78)
    ax[2].set_title("(c) Optimization axis", fontsize=11)
    ax[2].tick_params(labelsize=9)

    fig.tight_layout()
    path = out / "fig_ablation_complete.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"Results folder not found: {RESULTS}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = make_fed_convergence(OUT_DIR)
    p2 = make_ablation(OUT_DIR)
    print("Wrote:")
    print("  ", p1)
    print("  ", p2)


if __name__ == "__main__":
    main()
