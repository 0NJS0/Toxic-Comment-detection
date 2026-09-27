#!/usr/bin/env python3
"""
MODULE 0: Offline Teacher Pass (Llama-3.2-3B -> soft targets + spans)
======================================================================

Generates the evidence-anchored supervision for the whole project:

    1. Loads cleaned data from Module 1.
    2. Subsamples (all toxic + balanced clean) to ``llm.subsample_size``.
    3. Runs the teacher LLM on each comment (span-justification prompt).
    4. Caches per-comment: 6-label soft targets, evidence spans, anchor char.
    5. Builds the typing-faithful prefix bank (re-tokenized char prefixes).

Resume-safe: each finished comment is written to a Parquet shard; re-running
the module only processes comments not yet cached. Guarded by a wall-clock
budget (``llm.max_minutes``) for intermittent Kaggle sessions.

Run:
    uv run python module_00_teacher.py                 # configs/config.yaml
    uv run python module_00_teacher.py configs/other.yaml --mock   # smoke test

Outputs
-------
- results/llm_cache/*.parquet      (teacher soft targets + spans + anchors)
- results/prefix_cache/*.parquet   (re-tokenized char prefixes)
"""

import argparse
import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.helpers import set_random_seed, ensure_dir
from llm.teacher import run_teacher_pass


def main():
    parser = argparse.ArgumentParser(description="Offline LLM teacher pass.")
    parser.add_argument("config", nargs="?", default="configs/config.yaml")
    parser.add_argument("--mock", action="store_true",
                        help="Use a deterministic mock teacher (CPU smoke test).")
    parser.add_argument("--output", default="results/llm/teacher_summary.json")
    args = parser.parse_args()

    print("=" * 60)
    print("MODULE 0: TEACHER PASS (evidence-anchored supervision)")
    print("=" * 60)

    config = load_config(args.config)
    if args.mock:
        config["llm"]["mock"] = True
        config["llm"]["subsample_size"] = min(
            int(config["llm"].get("subsample_size", 100)), 200
        )

    set_random_seed(config["project"]["seed"])
    ensure_dir(config["results"]["logs"])

    summary = run_teacher_pass(config)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved: {out}")

    print(f"\n{'='*60}")
    print("MODULE 0 COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())