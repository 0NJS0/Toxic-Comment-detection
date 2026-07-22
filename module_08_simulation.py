#!/usr/bin/env python3
"""
===============================================================================
MODULE 8: Simulation UI (Streamlit)
===============================================================================

Interactive web demo for while-typing toxicity detection.

Run:
    uv run streamlit run simulation/app.py
===============================================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("FedPref: Simulation UI (Module 8)")
print("=" * 60)
print()
print("Start the Streamlit app with:")
print()
print("    uv run streamlit run simulation/app.py")
print()
print("Or use the helper:")
print()
print("    python module_08_simulation.py --run")
print()

import argparse
parser = argparse.ArgumentParser(description="FedPref Simulation UI")
parser.add_argument("--run", action="store_true", help="Launch Streamlit app")
args = parser.parse_args()

if args.run:
    import subprocess
    subprocess.run(["uv", "run", "streamlit", "run", "simulation/app.py"], check=True)
