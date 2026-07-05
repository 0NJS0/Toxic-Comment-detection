#!/usr/bin/env python3
"""
===============================================================================
MODULE 10: Reports and Plots
===============================================================================

Consolidates all experiment results into:
  - Training curves (PNG)
  - Prefix evaluation plot (PNG)
  - Ablation comparison plots (PNG per axis)
  - Ablation summary table (Markdown)
  - Final report (Markdown)

Run:
    uv run python module_10_reports.py
===============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from evaluation.reports import generate_all


def main():
    print("=" * 60)
    print("FedPref: Reports & Plots (Module 10)")
    print("=" * 60)

    generate_all("results/reports")

    print(f"\n{'='*60}")
    print("MODULE 10 COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
