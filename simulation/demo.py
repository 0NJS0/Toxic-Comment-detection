"""
Notebook/CLI Demo (replaces the Streamlit UI inside Kaggle notebooks)
======================================================================

Streamlit cannot run interactively in a Kaggle notebook, so this module
provides the same *while-typing early detection* showcase as the Streamlit
app (`simulation/app.py`) as plain printed output plus a matplotlib bar
chart. It uses the trained DistilBERT classifier from Module 3/5 and shows
per-label probabilities at the full text and several prefix lengths.

Usage:
    from simulation.demo import run_demo
    run_demo()
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch


def find_best_checkpoint(ckpt_dir: str = "results/checkpoints") -> str:
    """Return path to the best available trained model.

    Prefers a `.pt` epoch checkpoint (loadable by `load_predictor`), else
    the best_model pretrained directory.
    """
    ckpt_dir = Path(ckpt_dir)
    epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                         key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if epoch_ckpts:
        return str(epoch_ckpts[-1])
    if (ckpt_dir / "best_model").exists():
        return str(ckpt_dir / "best_model")
    raise FileNotFoundError("No trained model found. Run Module 3 (or 5) first.")


def run_demo(
    config_path: str = "configs/config.yaml",
    checkpoint_path: str = None,
    examples=None,
):
    import matplotlib.pyplot as plt
    import torch
    from utils.config import load_config
    from simulation.predictor import ToxicityPredictor
    from models.distilbert import DistilBERTForMultiLabelClassification
    from transformers import AutoTokenizer

    config = load_config(config_path)
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(config["results"]["checkpoints"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a predictor from either the best dir or a raw epoch checkpoint.
    ckpt = Path(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(config["data"]["processed"]["path"]) / "tokenized" / "tokenizer")
    )
    if ckpt.is_dir() and (ckpt / "head_config.pt").exists():
        model = DistilBERTForMultiLabelClassification.from_pretrained(
            str(ckpt), model_name=config["model"]["name"],
        )
    else:
        from simulation.predictor import load_predictor
        predictor = load_predictor(checkpoint_path, config_path=config_path, device=device)
        model = None

    if model is not None:
        predictor = ToxicityPredictor(model, tokenizer, device=device)

    if examples is None:
        examples = [
            "I love this! Great work!",
            "You are an idiot and a moron.",
            "Go kill yourself, you piece of trash.",
            "This is the worst thing I've ever seen.",
            "I respectfully disagree with your point.",
        ]

    print("=" * 70)
    print("FedPref while-typing demo (notebook version)")
    print(f"Model: {config['model']['name']} | Device: {str(predictor.device).upper()}")
    print("=" * 70)

    for idx, text in enumerate(examples):
        print(f"\n--- Example {idx+1}: {text!r} ---")
        full = predictor.predict(text)
        mark = "TOXIC !!!" if full["is_toxic"] else "ok -"
        print(f"  full text: {mark}  max_prob={full['max_probability']:.3f}")
        print(f"  {'prefix':<8}{'tox?':<6}{'max_prob':<10}hot labels")
        for plen in [8, 16, 32, 64]:
            pre = predictor.predict_prefix(text, plen)
            hot = ", ".join(k for k, v in pre["binary"].items() if v) or "-"
            print(f"  {str(pre['prefix_length_tokens']):<8}"
                  f"{('yes' if pre['is_toxic'] else 'no '):<6}"
                  f"{pre['max_probability']:<10.4f}{hot}")

    # Bar chart for the last example (full-text probabilities)
    fig, ax = plt.subplots(figsize=(8, 3))
    r = predictor.predict(examples[-1])
    labels = ToxicityPredictor.LABEL_NAMES
    probs = [r["probabilities"][k] for k in labels]
    colors = ["#e74c3c" if p >= predictor.threshold else "#2ecc71" for p in probs]
    ax.bar(labels, probs, color=colors)
    ax.axhline(predictor.threshold, ls="--", color="#888", label="threshold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability")
    ax.set_title(f"Per-label toxicity: {examples[-1]!r}")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()