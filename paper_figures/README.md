# Paper figures

Reproducible generation of the two figures that were added to the ATAM-FET
manuscript from the federated full-cycle run.

## What this produces

| Figure file | Used in paper as | Plotted from (tracked in git) |
|---|---|---|
| `fig_fed_convergence4.png` | Federated convergence over 25 rounds (IID, Dirichlet 0.5, Dirichlet 0.1, Topic) | `Atam_Fet Run without ATAM Federated/results/federated/all_experiments_summary.json` |
| `fig_ablation_complete.png` | Complete 4-axis ablation (ATAM / prefix length / optimization) | `.../results/ablation/prefix_summary.json`, `.../results/ablation/experiments.json`, `.../results/optimization/benchmark_results.json` |

Every number drawn is read at runtime from the JSON result files listed
above; nothing is hard-coded from memory. Re-running the script regenerates
the exact PNGs shipped with the paper, so the code plus its inputs are the
reproducibility proof.

## Run

```bash
python paper_figures/generate_paper_figures.py
```

Outputs are written into this folder at 300 dpi. Requires `matplotlib`
(already in the project environment).

## Provenance

The source run is recorded in
`Atam_Fet Run without ATAM Federated/results/environment.json`
(seed 42, PyTorch 2.10 + CUDA 12.8, Tesla T4, Flower 1.35, transformers 5.0).
The committed PNGs match the script output byte-for-byte (verified by
SHA-256).
