#!/usr/bin/env python3
"""
Build the self-contained Kaggle Jupyter notebook for ATAM-FET.

Reads every source file in the repo, embeds each as a `%%writefile` cell,
and adds run cells that execute the module mains in dependency order.

Output: ATAM_FET_kaggle.ipynb   (intended to run in /kaggle/working)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ATAM_FET_kaggle.ipynb"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src.rstrip("\n") + "\n"}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.rstrip("\n") + "\n"}


WRITE_TARGETS = []


def raw_file(rel_path: str) -> dict:
    content = (ROOT / rel_path).read_text()
    WRITE_TARGETS.append(rel_path)
    return code(f"%%writefile {rel_path}\n" + content)


def batch(files, then="", name=""):
    """A markdown header + one writefile cell per file + an optional run cell."""
    cells = [md(f"#### {name}\nWrites:\n" + "\n".join(f"`{f}`" for f in files))]
    cells += [raw_file(f) for f in files]
    if then:
        cells.append(code(then))
    return cells


cells = []

# ---------------------------------------------------------------- intro
cells.append(md(
    "# ATAM-FET — Federated Edge-Based Early Toxicity Detection for Mobile Keyboards\n"
    "Jigsaw multi-label toxic-comment classification pipeline in a single, self-contained\n"
    "Kaggle notebook. All repo source is written into the working directory by the\n"
    "`%%writefile` cells, so the code is identical to the repository.\n"
    "\n"
"| Mod | Purpose | Heavy? |\n"
    "|-----|---------|--------|\n"
    "| 0 | **LLM teacher pass** (soft targets + evidence spans + prefix bank) | GPU heavy (LLM) |\n"
    "| 1 | Data cleaning | light |\n"
    "| 2 | Tokenization (80/10/10 stratified split) | light |\n"
    "| 3 | Centralized DistilBERT training (baseline) | GPU heavy |\n"
    "| 4 | LoRA personalization check | light |\n"
    "| 5 | Federated learning (FedAvg + LoRA) | heavy |\n"
    "| 6 | ATAM — attention module training | GPU heavy |\n"
    "| 7 | INT8 quantization + pruning + benchmark | light |\n"
    "| 8 | While-typing detection demo | light |\n"
    "| 9 | Ablation study (REAL numbers) | medium |\n"
    "| 10 | Reports & plots | light |\n"
    "\n"
    "Run the cells top-to-bottom. Each module saves outputs under `results/`, so a\n"
    "heavy cell can be interrupted and its outputs reused by later cells. Heavy GPU\n"
    "work (Module 0, 3, 5) is split into *session-paced* cells with wall-clock\n"
    "budgets (see the Module 0 section) and cross-session snapshot/restore.\n"
))

# ---------------------------------------------------------------- setup
cells.append(md(
    "## 0) Setup\n"
    "Install the two packages the pipeline needs (`emoji` for cleaning, `flwr` for the\n"
    "federated client base type). Everything else ships with Kaggle kernels."
))
cells.append(code(
    "!pip -q install -U emoji flwr bitsandbytes accelerate\n"
    "import os, sys, shutil\n"
    "from pathlib import Path\n"
    "\n"
    "os.chdir('/kaggle/working')\n"
    "for d in ['configs','data/raw','data/processed/tokenized','results',\n"
    "          'models/cache','training','preprocessing','evaluation',\n"
    "          'federated','optimization','simulation','utils']:\n"
    "    Path(d).mkdir(parents=True, exist_ok=True)\n"
    "print('workdir:', Path.cwd())\n"
    "print('python :', sys.version.split()[0])"
))

cells.append(md(
    "### 0.1) Put the raw Jigsaw CSVs in place\n"
    "The pipeline reads `data/raw/train.csv` (`test.csv` is optional - it has no\n"
    "labels anyway, so every module derives train/validation/test splits from the\n"
    "**train** file).\n"
    "- Either upload `train.csv` via **File → Upload** into `data/raw/`, **or**\n"
    "- set `RAW_DATA_DIR` to a Kaggle dataset and this cell copies the file(s),\n"
    "  so all downstream modules stay untouched."
))
cells.append(code(
    "RAW_DATA_DIR = Path('/kaggle/input/datasets/najmus07/jigasaw-train')  # contains train.csv\n"
    "\n"
    "targets = [RAW_DATA_DIR, Path('/kaggle/working/data/raw')]\n"
    "ok = False\n"
    "for src in (p for p in targets if p is not None):\n"
    "    if (src/'train.csv').exists():\n"
    "        shutil.copy(src/'train.csv', 'data/raw/train.csv')\n"
    "        if (src/'test.csv').exists():\n"
    "            shutil.copy(src/'test.csv', 'data/raw/test.csv')\n"
    "        else:\n"
    "            print('note: no test.csv found - train-only mode (val/test derived from train)')\n"
    "        print('raw data ready from', src)\n"
    "        ok = True\n"
    "        break\n"
    "if not ok:\n"
    "    print('Please put train.csv into data/raw/ (or fix RAW_DATA_DIR) and re-run this cell.')"
))

# ---------------------------------------------------------------- config
cells.append(md("## Shared config (all modules read this)"))
cells.append(raw_file("configs/config.yaml"))

# ---------------------------------------------------------------- module 1
cells.append(md(
    "## Module 1 — Data Preparation\n"
    "Loads `data/raw`, cleans text (URLs / mentions / emoji→text / lowercase /\n"
    "whitespace), removes duplicates, computes stats.\n"
    "**Output:** `data/processed/train_cleaned.csv` + `test_cleaned.csv`."
))
cells += batch(
    ["utils/__init__.py", "utils/config.py", "utils/helpers.py",
     "preprocessing/__init__.py", "preprocessing/clean.py",
     "module_01_data_preparation.py"],
    name="Module 1",
)
cells.append(code("!python module_01_data_preparation.py"))

# ---------------------------------------------------------------- module 2
cells.append(md(
    "## Module 2 — Tokenization\n"
    "80/10/10 stratified split, DistilBERT tokenizer, saves the arrow dataset.\n"
    "**Output:** `data/processed/tokenized/` (train/val/test + tokenizer + stats)."
))
cells += batch(["preprocessing/tokenize.py", "module_02_tokenization.py"], name="Module 2")
cells.append(code("!python module_02_tokenization.py"))

# ---------------------------------------------------------------- module 0
cells.append(md(
    "## Module 0 — Offline LLM Teacher Pass (session-1 heavy step)\n"
    "Runs Llama-3.2-3B once over a **stratified subsample** (all toxic + balanced\n"
    "clean) and, per comment, produces the evidence-anchored supervision the rest\n"
    "of the research consumes:\n"
    "  - **soft targets**  — 6-label confidence vector (the teacher's final decision)\n"
    "  - **evidence spans** — character offsets justifying each label\n"
    "  - **anchor char**   — earliest position the decision becomes knowable\n"
    "  - **prefix bank**   — re-tokenized char prefixes (typing-faithful)\n"
    "\n"
    "**Session pacing (~5-6 h GPU):** this is the single heaviest step. The cells\n"
    "Recompose below stop cleanly at `llm.max_minutes` (default 300). If the session\n"
    "is interrupted, re-run this cell (or the next session after restoring the\n"
    "snapshot) and it resumes from the Parquet cache — finished comments are never\n"
    "re-processed. Add `--mock` for a 2-minute CPU smoke test.\n"
    "\n"
    "**Resume across sessions:** every heavy module ends with a snapshot cell that\n"
    "publishes `results/` as a Kaggle Dataset (`sessions.snapshot_slug`); the top of\n"
    "a new session automatically restores it. If the `kaggle` CLI/credentials are\n"
    "missing, the same tarball + `kaggle datasets` commands are printed for manual\n"
    "upload.\n"
    "\n"
    "| Session | Cells | Est. time |\n"
    "|---------|-------|-----------|\n"
    "| S1 | Modules 1, 2, 0 | 5-6 h |\n"
    "| S2 | Modules 3 (baseline + anchored), re-run eval | 4-5 h |\n"
    "| S3 | Module 5 FL + DP sweep | 7-8 h |\n"
    "| S4 | Module 5 topic run, MIA, 9, 10 | 4-5 h |\n"
    "\n"
    "**Outputs:** `results/llm_cache/` + `results/prefix_cache/`."
))
cells += batch(
    ["llm/__init__.py", "llm/prompts.py", "llm/spans.py", "llm/cache.py",
     "llm/teacher.py", "preprocessing/anchors.py", "utils/kaggle_utils.py",
     "module_00_teacher.py"],
    name="Module 0 core",
)
cells.append(code(
    "import json\n"
    "from pathlib import Path\n"
    "from utils.config import load_config\n"
    "from utils.kaggle_utils import restore_from_dataset, snapshot_dataset\n"
    "\n"
    "def _cache_ready(config):\n"
    "    return Path(config['llm']['output_dir']).exists() and bool(\n"
    "        list(Path(config['llm']['output_dir']).glob('part-*.parquet')))\n"
    "\n"
    "config = load_config('configs/config.yaml')\n"
    "ready = _cache_ready(config)\n"
    "\n"
    "# Cross-session resume: pull a previous snapshot into the working dir.\n"
    "if not ready and config['sessions'].get('snapshot'):\n"
    "    restore_from_dataset('.', config['sessions'].get('snapshot_slug'))\n"
    "    ready = _cache_ready(config)\n"
    "\n"
    "if ready:\n"
    "    print('Teacher cache already present - skipping (re-run not needed).')\n"
    "else:\n"
    "    get_ipython().system('python module_00_teacher.py')\n"
    "\n"
    "done = _cache_ready(config)\n"
    "if done and config['sessions'].get('snapshot'):\n"
    "    snapshot_dataset(['results/llm_cache', 'results/prefix_cache'],\n"
    "                     config['sessions'].get('snapshot_slug'),\n"
    "                     message='module0 teacher pass', base_dir='.')\n"
    "print('MODULE 0 ready:', done)\n"
))

# ---------------------------------------------------------------- module 3
cells.append(md(
    "## Module 3 — Centralized DistilBERT Training\n"
    "Trains the baseline: `[CLS]` → dropout → 6-logit head, BCEWithLogitsLoss with\n"
    "`pos_weight`, AdamW + linear warmup, early stopping, per-epoch checkpoints.\n"
    "This and Module 6 are the most expensive cells (interrupt to continue later;\n"
    "`results/checkpoints/*` persist).\n"
    "**Output:** `results/checkpoints`, `results/logs`, `results/plots`."
))
cells += batch(
    ["evaluation/__init__.py", "evaluation/metrics.py",
     "models/__init__.py", "models/distilbert.py", "models/atam.py",
     "models/lora.py", "models/registry.py",
     "training/__init__.py", "training/train.py",
     "module_03_training.py"],
    name="Module 3 core",
)
cells.append(code("!python module_03_training.py"))
cells.append(code(
    "from utils.config import load_config\n"
    "from utils.kaggle_utils import snapshot_dataset\n"
    "_c = load_config('configs/config.yaml')\n"
    "if _c['sessions'].get('snapshot'):\n"
    "    snapshot_dataset(['results/checkpoints', 'results/logs', 'results/plots'],\n"
    "                     _c['sessions'].get('snapshot_slug'),\n"
    "                     message='module3 baseline', base_dir='.')\n"
))

# ---------------------------------------------------------------- module 4
cells.append(md(
    "## Module 4 — LoRA Personalization\n"
    "Frozen encoder + low-rank adapters on `q_lin`/`v_lin`. Prints trainable %.\n"
    "This is the parameter budget the federated step actually communicates."
))
cells += batch(["module_04_lora.py"], name="Module 4")
cells.append(code("!python module_04_lora.py"))

# ---------------------------------------------------------------- module 5
cells.append(md(
    "## Module 5 — Federated Learning (FedAvg + LoRA)\n"
    "Partitions data (IID / Dirichlet / Topic), creates one client per partition,\n"
    "trains locally for `local_epochs`, aggregates via FedAvg, tracks the global\n"
    "validation curve. Only LoRA + classifier-head params are exchanged.\n"
    "\n"
    "**Budget it!** A full paper sweep (4 partition schemes × 25 rounds × 10 clients)\n"
    "can exceed a single session. The cell below starts with IID + a few rounds;\n"
    "extend `EXPERIMENTS` and `ROUNDS` as compute allows."
))
cells += batch(
    ["federated/__init__.py", "federated/partition.py", "federated/client.py",
     "federated/server.py", "federated/strategies.py",
     "federated/train_federated.py"],
    name="Module 5 core",
)
cells.append(code(
    "import time, json\n"
    "from pathlib import Path\n"
    "from utils.config import load_config\n"
    "from federated.train_federated import run_experiment\n"
    "\n"
    "# ---------- federated budget ----------\n"
    "ROUNDS     = 5        # paper uses 25\n"
    "CLIENTS    = 10\n"
    "LOCAL_EP   = 1\n"
    "EXPERIMENTS = [('iid', 'iid', None), ('dirichlet', 'dirichlet', 0.5)]\n"
    "# add 'topic' partition and alpha=0.1 for the full sweep\n"
    "# --------------------------------------\n"
    "\n"
    "config = load_config('configs/config.yaml')\n"
    "config['federated']['num_rounds'] = ROUNDS\n"
    "config['federated']['num_clients'] = CLIENTS\n"
    "config['federated']['local_epochs'] = LOCAL_EP\n"
    "\n"
    "for name, ptype, alpha in EXPERIMENTS:\n"
    "    config['federated']['partition_type'] = ptype\n"
    "    config['federated']['iid'] = (ptype == 'iid')\n"
    "    if alpha is not None:\n"
    "        config['federated']['alpha'] = alpha\n"
    "\n"
    "    t0 = time.time()\n"
    "    print(f'\\n########## EXPERIMENT: {name} ##########')\n"
    "    res = run_experiment(config)\n"
    "    out = Path('results/federated/experiment_' + name + '.json')\n"
    "    out.write_text(json.dumps(res, indent=2, default=str))\n"
    "    print(f'saved -> {out}   ({time.time()-t0:.0f}s)')\n"
    "\n"
    "# ---- snapshot for cross-session resume ----\n"
    "from utils.kaggle_utils import snapshot_dataset\n"
    "if config['sessions'].get('snapshot'):\n"
    "    snapshot_dataset(['results/federated'], config['sessions'].get('snapshot_slug'),\n"
    "                     message='module5 federated', base_dir='.')\n"
))

# ---------------------------------------------------------------- module 6
cells.append(md(
    "## Module 6 — ATAM (Adaptive Toxic Attention)\n"
    "DistilBERT + learnable attention parser over token states; trains to the\n"
    "same schedule as Module 3, evaluates on test with confusion matrices.\n"
    "**Output:** `results/atam/`."
))
cells += batch(["module_06_atam.py"], name="Module 6")
cells.append(code("!python module_06_atam.py"))

# ---------------------------------------------------------------- module 7
cells.append(md(
    "## Module 7 — Model Optimization\n"
    "INT8 dynamic quantization + 30% magnitude pruning of the Module-3 model,\n"
    "benchmarked (latency/size/params). Runs best on CPU."
))
cells += batch(
    ["optimization/__init__.py", "optimization/quantize.py", "optimization/prune.py",
     "optimization/benchmark.py", "optimization/distill.py",
     "module_07_optimization.py"],
    name="Module 7 core",
)
cells.append(code("!python module_07_optimization.py"))

# ---------------------------------------------------------------- module 8
cells.append(md(
    "## Module 8 — While-Typing Simulation (notebook demo)\n"
    "Streamlit can't run inside a Kaggle notebook, so this replaces the UI with\n"
    "a text demo: per-label probabilities at the full text and at prefixes of\n"
    "8/16/32/64 tokens (early detection)."
))
cells += batch(
    ["simulation/__init__.py", "simulation/predictor.py", "simulation/demo.py",
     "module_08_simulation.py"],
    name="Module 8 core",
)
cells.append(code(
    "from simulation.demo import run_demo\n"
    "run_demo()"
))

# ---------------------------------------------------------------- module 9
cells.append(md(
    "## Module 9 — Ablation Study (REAL metrics)\n"
    "Short fine-tunes and real prefix/quantize/prune evaluations across the ATAM,\n"
    "Prefix, Optimization axes; FedAvg rows read `results/federated/` history if\n"
    "Module 5 already ran. **Output:** `results/ablation/experiments.json`."
))
cells += batch(["evaluation/ablation.py", "module_09_ablation.py"], name="Module 9")
cells.append(code("!python module_09_ablation.py"))

# ---------------------------------------------------------------- module 10
cells.append(md(
    "## Module 10 — Reports & Plots\n"
    "Generates the final markdown report, training curves and bar charts.\n"
    "**Output:** `results/reports/`."
))
cells += batch(["evaluation/reports.py", "module_10_reports.py"], name="Module 10")
cells.append(code("!python module_10_reports.py"))

cells.append(md("## Wrap up\n"
"Teacher cache: `results/llm_cache/` + prefix bank: `results/prefix_cache/` ·\n"
"Checkpoints: `results/checkpoints/` · Federated history: `results/federated/` ·\n"
"Report: `results/reports/final_report.md` · Ablation: `results/ablation/experiments.json`.\n"
"\n"
"**Next session:** attach the `sessions.snapshot_slug` Kaggle dataset (auto-created\n"
"by the snapshot cells) and run from the Module you stopped at — inputs restore\n"
"into `results/` automatically via `restore_from_dataset`."))

# ------------------------------------------------------------------ setup dirs
# Auto-derive the setup cell's mkdir list from the actual writefile targets so a
# future added file (or removed dir) can never break %%writefile again.
runtime_dirs = {"data/raw", "data/processed/tokenized", "results", "models/cache"}
write_parents = {str(Path(t).parent) for t in WRITE_TARGETS}
write_parents.discard(".")  # the working dir always exists
dirs = sorted(write_parents | runtime_dirs)

missing = write_parents - set(dirs)
assert not missing, f"setup mkdir list would miss %%writefile parent dirs: {sorted(missing)}"

setup_src = (
    "!pip -q install -U emoji flwr bitsandbytes accelerate\n"
    "import os, sys, shutil\n"
    "from pathlib import Path\n"
    "\n"
    "os.chdir('/kaggle/working')\n"
    "for d in " + repr(dirs) + ":\n"
    "    Path(d).mkdir(parents=True, exist_ok=True)\n"
    "print('workdir:', Path.cwd())\n"
    "print('python :', sys.version.split()[0])"
)
for c in cells:
    if c.get("cell_type") == "code" and "os.chdir('/kaggle/working')" in c["source"]:
        c["source"] = setup_src
        break
else:
    raise RuntimeError("setup cell not found to patch")

notebook = {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {OUT}  ({len(cells)} cells)")
print(f"setup mkdir dirs: {dirs}")