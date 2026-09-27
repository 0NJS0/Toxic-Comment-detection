"""
Ablation Study (REAL metrics)
=============================

Replaces the earlier placeholder that returned random numbers with a
computation of genuine, reproducible ablation results:

  * ATAM axis          - real short fine-tune: DistilBERT vs DistilBERT + ATAM
  * Prefix axis        - real prefix truncation of the trained model:
                         full / 64 / 32 / 16 / 8 tokens
  * Federated axis     - centralized reference (short fine-tune) vs
                         actual FedAvg results saved by Module 5
  * Optimization axis  - full-precision vs INT8 dynamic-quantized
                         vs 30% magnitude-pruned model

To stay fast on Kaggle, the fine-tune axes use a small fixed subset and
a single epoch. Evaluation axes (prefix/optimization) reuse the best
classifier from Module 3.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader

# Protocol constants (cheap enough to run in a notebook)
ABLATION_TRAIN_SAMPLE = 2560
ABLATION_VAL_SAMPLE = 1024
ABLATION_EPOCHS = 1


@dataclass
class AblationResult:
    experiment_id: str
    axis: str
    variant: str
    f1_macro: float
    accuracy: float
    roc_auc: float
    trainable_params: int
    total_params: int
    training_time_s: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _collate(batch):
    input_ids = torch.tensor([s["input_ids"] for s in batch], dtype=torch.long)
    attention_mask = torch.tensor([s["attention_mask"] for s in batch], dtype=torch.long)
    labels = torch.tensor([s["labels"] for s in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _select(ds, size: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    size = min(size, len(ds))
    idx = rng.choice(len(ds), size=size, replace=False)
    return ds.select(idx.tolist())


def _count_params(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _eval_dataset(model, ds, device, batch_size: int = 32) -> Dict:
    """Evaluate a HF dataset through a model and return metric dict."""
    from evaluation.metrics import compute_metrics
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    logits_list, labels_list = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            logits_list.append(logits.cpu().numpy())
            labels_list.append(b["labels"].numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return compute_metrics(logits, labels)


def _eval_wrapper(model, ds, device, batch_size: int = 32) -> Dict:
    """Evaluate a model that is *not* invoked as model(input_ids, attention_mask)."""
    from evaluation.metrics import compute_metrics
    dev = torch.device("cpu")  # quantized INT8 models are CPU-only
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    logits_list, labels_list = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev))
            logits = out.logits if hasattr(out, "logits") else out
            logits_list.append(logits.cpu().numpy())
            labels_list.append(b["labels"].numpy())
    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return compute_metrics(logits, labels)


class AblationRunner:
    def __init__(self, base_config: dict, device: torch.device):
        self.base = base_config
        self.device = device
        self.label_cols = base_config["data"]["labels"]
        self.results: List[AblationResult] = []

    # ------------------------------------------------------------------ #
    # 1. ATAM axis: real short fine-tune
    # ------------------------------------------------------------------ #
    def run_atam_ablation(self):
        from datasets import load_from_disk
        ds = load_from_disk(str(Path(self.base["data"]["processed"]["path"]) / "tokenized"))

        train_sub = _select(ds["train"], ABLATION_TRAIN_SAMPLE, seed=1)
        val_sub = _select(ds["validation"], ABLATION_VAL_SAMPLE, seed=2)

        for vid, label, use_atam in [
            ("without_atam", "DistilBERT (no ATAM)", False),
            ("with_atam", "DistilBERT + ATAM", True),
        ]:
            metrics, tt, trainable, total = self._finetune(
                train_sub, val_sub, use_atam=use_atam, epochs=ABLATION_EPOCHS,
            )
            self.results.append(AblationResult(
                experiment_id=f"atam_{vid}", axis="ATAM", variant=label,
                f1_macro=metrics["f1_macro"], accuracy=metrics["accuracy"],
                roc_auc=metrics["roc_auc"], trainable_params=trainable,
                total_params=total, training_time_s=tt, notes="real subset fine-tune",
            ))

    # ------------------------------------------------------------------ #
    # 2. Prefix (early detection) axis: real truncated inputs
    # ------------------------------------------------------------------ #
    def run_prefix_ablation(self):
        from datasets import load_from_disk
        from evaluation.metrics import compute_metrics

        ds = load_from_disk(str(Path(self.base["data"]["processed"]["path"]) / "tokenized"))
        model = self._load_best_classifier()
        device = self.device
        val_sub = _select(ds["validation"], ABLATION_VAL_SAMPLE, seed=2)

        ids = torch.tensor(val_sub["input_ids"], dtype=torch.long)
        mask = torch.tensor(val_sub["attention_mask"], dtype=torch.long)
        labels = np.array(val_sub["labels"], dtype=np.float32)

        full_logits = self._batch_forward(model, ids, mask)
        full_bin = full_logits >= 0

        for vid, length in [("full", None), ("prefix_64", 64), ("prefix_32", 32),
                            ("prefix_16", 16), ("prefix_8", 8)]:
            if length is None:
                p_ids, p_mask = ids, mask
            else:
                length = min(length, ids.shape[1])
                p_ids, p_mask = ids[:, :length], mask[:, :length]
            logs = self._batch_forward(model, p_ids, p_mask)
            metrics = compute_metrics(logs, labels)
            agreement = float(np.mean((logs >= 0) == full_bin))
            self.results.append(AblationResult(
                experiment_id=f"prefix_{vid}", axis="Prefix Length",
                variant="Full text" if length is None else f"Prefix {length} tokens",
                f1_macro=metrics["f1_macro"], accuracy=metrics["accuracy"],
                roc_auc=metrics["roc_auc"], trainable_params=0, total_params=0,
                training_time_s=0.0,
                notes=f"agree vs full = {agreement:.3f}",
            ))

    # ------------------------------------------------------------------ #
    # 3. Federated axis: real FL history (from Module 5) + central reference
    # ------------------------------------------------------------------ #
    def run_federated_ablation(self):
        from datasets import load_from_disk
        ds = load_from_disk(str(Path(self.base["data"]["processed"]["path"]) / "tokenized"))
        train_sub = _select(ds["train"], ABLATION_TRAIN_SAMPLE, seed=42)
        val_sub = _select(ds["validation"], ABLATION_VAL_SAMPLE, seed=2)

        # Centralized reference (real short fine-tune)
        metrics, tt, trainable, total = self._finetune(
            train_sub, val_sub, use_atam=False, epochs=ABLATION_EPOCHS,
        )
        self.results.append(AblationResult(
            experiment_id="fed_centralized", axis="Federated Rounds",
            variant="Centralized", f1_macro=metrics["f1_macro"],
            accuracy=metrics["accuracy"], roc_auc=metrics["roc_auc"],
            trainable_params=trainable, total_params=total, training_time_s=tt,
            notes="real subset fine-tune",
        ))

        # Actual federated numbers if Module 5 has run
        hist_path = Path("results/federated/federated_history.json")
        if hist_path.exists():
            with open(hist_path) as f:
                hit = json.load(f)
            rounds = hit.get("history", {}).get("round", [])
            f1s = hit.get("history", {}).get("global_val_f1_macro", [])
            accs = hit.get("history", {}).get("global_val_accuracy", [])
            for want, vid in [(10, "fed_10"), (25, "fed_25")]:
                if rounds and want <= max(rounds) and want - 1 < len(f1s):
                    i = min(want - 1, len(f1s) - 1)
                    self.results.append(AblationResult(
                        experiment_id=vid, axis="Federated Rounds",
                        variant=f"FedAvg {want} rounds",
                        f1_macro=f1s[i] if i < len(f1s) else 0.0,
                        accuracy=accs[i] if i < len(accs) else 0.0,
                        roc_auc=0.0, trainable_params=0, total_params=0,
                        training_time_s=0.0, notes="from Module 5 history",
                    ))
        else:
            print("  [hint] No results/federated history yet — run Module 5 "
                  "to populate the FedAvg rows.")

    # ------------------------------------------------------------------ #
    # 4. Optimization axis: real quantize / prune on trained model
    # ------------------------------------------------------------------ #
    def run_optimization_ablation(self):
        from datasets import load_from_disk
        from optimization.quantize import quantize_model_dynamic
        from optimization.prune import prune_model_magnitude

        ds = load_from_disk(str(Path(self.base["data"]["processed"]["path"]) / "tokenized"))
        val_sub = _select(ds["validation"], ABLATION_VAL_SAMPLE, seed=2)
        device = self.device

        model_fp32 = self._load_best_classifier()
        metrics = _eval_dataset(model_fp32, val_sub, device)
        self.results.append(AblationResult(
            experiment_id="opt_full_precision", axis="Optimization",
            variant="FP32", f1_macro=metrics["f1_macro"],
            accuracy=metrics["accuracy"], roc_auc=metrics["roc_auc"],
            trainable_params=0, total_params=0, training_time_s=0.0,
            notes="eval on best Class-3 model",
        ))

        qmodel = model_fp32.cpu()
        qwrap = quantize_model_dynamic(qmodel, dtype=torch.qint8)
        metrics_q = _eval_wrapper(qwrap, val_sub, device)
        self.results.append(AblationResult(
            experiment_id="opt_quantized", axis="Optimization",
            variant="INT8 Quantized", f1_macro=metrics_q["f1_macro"],
            accuracy=metrics_q["accuracy"], roc_auc=metrics_q["roc_auc"],
            trainable_params=0, total_params=0, training_time_s=0.0,
            notes="dynamic INT8",
        ))

        pmod = self._load_best_classifier()
        prune_model_magnitude(pmod, amount=0.3, layers_to_prune="linear", make_permanent=True)
        metrics_p = _eval_dataset(pmod, val_sub, device)
        self.results.append(AblationResult(
            experiment_id="opt_pruned_30", axis="Optimization",
            variant="Pruned 30%", f1_macro=metrics_p["f1_macro"],
            accuracy=metrics_p["accuracy"], roc_auc=metrics_p["roc_auc"],
            trainable_params=0, total_params=0, training_time_s=0.0,
            notes="magnitude pruned 30% (linear)",
        ))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _finetune(self, train_ds, val_ds, use_atam, epochs):
        """Short real fine-tune on a subset. Returns (metrics, trainable, total)."""
        from models.registry import create_model
        model = create_model(self.base, use_atam=use_atam)
        trainable, total = _count_params(model)
        model = model.to(self.device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.base["training"]["learning_rate"])
        lossfn = nn.BCEWithLogitsLoss()
        loader = DataLoader(train_ds, batch_size=self.base["training"]["batch_size"],
                            shuffle=True, collate_fn=_collate)
        t0 = time.time()
        model.train()
        for _ in range(epochs):
            for b in loader:
                ids = b["input_ids"].to(self.device)
                mask = b["attention_mask"].to(self.device)
                labels = b["labels"].to(self.device)
                opt.zero_grad()
                loss = lossfn(model(ids, mask), labels)
                loss.backward()
                opt.step()
        tt = time.time() - t0
        metrics = _eval_dataset(model, val_ds, self.device)
        return metrics, tt, trainable, total

    def _load_best_classifier(self, device=None):
        """Best centrally trained model: last epoch checkpoint, else best_model dir."""
        from models.distilbert import DistilBERTForMultiLabelClassification
        device = device or self.device
        ckpt_dir = Path(self.base["results"]["checkpoints"])
        epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                             key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if epoch_ckpts:
            ckpt = torch.load(str(epoch_ckpts[-1]), map_location="cpu", weights_only=True)
            model = DistilBERTForMultiLabelClassification(
                model_name=self.base["model"]["name"],
                num_labels=len(self.label_cols),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            return model.to(device)
        best_dir = ckpt_dir / "best_model"
        return DistilBERTForMultiLabelClassification.from_pretrained(
            str(best_dir), model_name=self.base["model"]["name"]).to(device)

    def _batch_forward(self, model, ids, mask, bs: int = 64):
        model.eval()
        device = self.device
        logits_list = []
        with torch.no_grad():
            for i in range(0, ids.size(0), bs):
                ids_b = ids[i:i + bs].to(device)
                mask_b = mask[i:i + bs].to(device)
                logits_list.append(model(ids_b, mask_b).cpu().numpy())
        return np.concatenate(logits_list, axis=0)

    # ------------------------------------------------------------------ #
    # Public orchestration
    # ------------------------------------------------------------------ #
    def run_all(self):
        self.run_atam_ablation()
        self.run_prefix_ablation()
        self.run_federated_ablation()
        self.run_optimization_ablation()
        return self.results

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"results": [asdict(r) for r in self.results]}, f, indent=2)
        print(f"\nSaved {len(self.results)} ablation results to {path}")

    def summary_table(self) -> str:
        lines = ["| Experiment | Axis | Variant | F1-macro | Acc | Params (M) | Notes |"]
        lines.append("|-----------|------|---------|----------|-----|-----------|-------|")
        for r in self.results:
            params = f"{r.total_params / 1e6:.2f}" if r.total_params else "-"
            lines.append(f"| {r.experiment_id} | {r.axis} | {r.variant} | "
                         f"{r.f1_macro:.4f} | {r.accuracy:.4f} | {params} | {r.notes} |")
        return "\n".join(lines)