"""
FedPref Ablation Study
=======================

Systematic comparison across 6 axes:

1. Model size: DistilBERT vs TinyBERT vs MobileBERT
2. LoRA rank: r=2, r=8, r=16, no LoRA (full fine-tune)
3. Prefix curriculum: with curriculum vs fixed prefix vs full text
4. Federated rounds: 10 vs 25 vs 50
5. Non-IID severity: Dirichlet α=0.1 vs 0.5 vs IID
6. Non-IID type: Dirichlet vs topic-based

Each experiment records: F1-macro, Accuracy, Latency, Model Size.
Results are saved to a JSON file for reporting.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


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
    model_size_mb: float
    notes: str = ""


class AblationRunner:
    """
    Runs ablation experiments by sweeping one configuration axis
    at a time while keeping others at their default.

    Usage:
        runner = AblationRunner(base_config, device)
        results = runner.run_all()
        runner.save("results/ablation/results.json")
    """

    def __init__(self, base_config: dict, device: torch.device):
        self.base = base_config
        self.device = device
        self.label_cols = base_config["data"]["labels"]
        self.results: List[AblationResult] = []

    def run_experiment(
        self,
        experiment_id: str,
        axis: str,
        variant: str,
        config_override: dict,
    ) -> AblationResult:
        """
        Run a single ablation experiment by merging config overrides.

        This is a template: subclasses override the actual training logic.
        """
        from copy import deepcopy
        cfg = deepcopy(self.base)
        for k, v in config_override.items():
            if k in cfg:
                cfg[k] = v
            else:
                # Support nested keys like "model.name"
                parts = k.split(".")
                target = cfg
                for p in parts[:-1]:
                    if p not in target:
                        target[p] = {}
                    target = target[p]
                target[parts[-1]] = v

        print(f"\n[{experiment_id}] {axis}={variant}")

        start_time = time.time()

        # --- Placeholder: actual training would go here ---
        # For the structured codebase, we'd typically:
        #   1. Load data
        #   2. Build model with overrides
        #   3. Train
        #   4. Evaluate

        # Simulated result (replace with actual training)
        dummy_f1 = 0.7 + np.random.uniform(-0.05, 0.05)
        dummy_acc = 0.93 + np.random.uniform(-0.01, 0.01)
        dummy_auc = 0.99 + np.random.uniform(-0.005, 0.005)

        # Compute trainable params
        from models.registry import create_model
        model = create_model(
            cfg,
            model_name=cfg.get("model", {}).get("name", "distilbert-base-uncased"),
            num_labels=len(self.label_cols),
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        elapsed = time.time() - start_time

        result = AblationResult(
            experiment_id=experiment_id,
            axis=axis,
            variant=variant,
            f1_macro=float(dummy_f1),
            accuracy=float(dummy_acc),
            roc_auc=float(dummy_auc),
            trainable_params=trainable,
            total_params=total,
            training_time_s=elapsed,
            model_size_mb=total * 4 / (1024 * 1024),
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------------ #
    # Experiment group: Model Size
    # ------------------------------------------------------------------ #
    def run_model_ablation(self):
        models = [
            ("distilbert-base-uncased", "DistilBERT (66M)"),
            ("huawei-noah/TinyBERT_General_4L_312D", "TinyBERT (14M)"),
            ("google/mobilebert-uncased", "MobileBERT (25M)"),
        ]
        for mname, label in models:
            self.run_experiment(
                f"model_{mname}", "model", label,
                {"model.name": mname},
            )

    # ------------------------------------------------------------------ #
    # Experiment group: LoRA Rank
    # ------------------------------------------------------------------ #
    def run_lora_ablation(self):
        ranks = [("none", "No LoRA"), ("2", "LoRA r=2"), ("8", "LoRA r=8"), ("16", "LoRA r=16")]
        for r, label in ranks:
            if r == "none":
                self.run_experiment(
                    f"lora_none", "lora_rank", label,
                    {"personalization.enabled": False},
                )
            else:
                self.run_experiment(
                    f"lora_r{r}", "lora_rank", label,
                    {"personalization.enabled": True, "personalization.lora_rank": int(r)},
                )

    # ------------------------------------------------------------------ #
    # Experiment group: Prefix Curriculum
    # ------------------------------------------------------------------ #
    def run_prefix_ablation(self):
        variants = [
            ("full", "Full text only", {"prefix.curriculum_epochs": 0, "prefix.prefix_lengths": [None]}),
            ("fixed_64", "Fixed prefix 64", {"prefix.curriculum_epochs": 0, "prefix.prefix_lengths": [64]}),
            ("curriculum", "Curriculum [8,16,32,64,128]", {"prefix.curriculum_epochs": 5, "prefix.prefix_lengths": [8, 16, 32, 64, 128]}),
        ]
        for vid, label, override in variants:
            self.run_experiment(
                f"prefix_{vid}", "prefix_strategy", label, override,
            )

    # ------------------------------------------------------------------ #
    # Experiment group: Federated Rounds
    # ------------------------------------------------------------------ #
    def run_fed_rounds_ablation(self):
        for rounds in [10, 25, 50]:
            self.run_experiment(
                f"fed_rounds_{rounds}", "fed_rounds", f"{rounds} rounds",
                {"federated.num_rounds": rounds},
            )

    # ------------------------------------------------------------------ #
    # Experiment group: Non-IID Severity
    # ------------------------------------------------------------------ #
    def run_noniid_ablation(self):
        variants = [
            ("iid", "IID", {"federated.partition_type": "iid"}),
            ("dirichlet_0.5", "Dirichlet α=0.5", {"federated.partition_type": "dirichlet", "federated.alpha": 0.5}),
            ("dirichlet_0.1", "Dirichlet α=0.1", {"federated.partition_type": "dirichlet", "federated.alpha": 0.1}),
            ("topic", "Topic-based", {"federated.partition_type": "topic"}),
        ]
        for vid, label, override in variants:
            self.run_experiment(
                f"noniid_{vid}", "non_iid_type", label, override,
            )

    # ------------------------------------------------------------------ #
    # Experiment group: Non-IID Type
    # ------------------------------------------------------------------ #
    def run_noniid_type_ablation(self):
        variants = [
            ("dirichlet_0.5", "Dirichlet α=0.5", {"federated.partition_type": "dirichlet", "federated.alpha": 0.5}),
            ("topic_10", "Topic (10 clusters)", {"federated.partition_type": "topic", "federated.num_clients": 10}),
            ("topic_20", "Topic (20 clusters)", {"federated.partition_type": "topic", "federated.num_clients": 20}),
        ]
        for vid, label, override in variants:
            self.run_experiment(
                f"noniid_type_{vid}", "non_iid_type", label, override,
            )

    def run_all(self):
        """Run all ablation experiments."""
        print("=" * 60)
        print("Ablation Study: 20+ Experiments")
        print("=" * 60)

        self.run_model_ablation()
        self.run_lora_ablation()
        self.run_prefix_ablation()
        self.run_fed_rounds_ablation()
        self.run_noniid_ablation()
        self.run_noniid_type_ablation()

        return self.results

    def save(self, path: str):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"results": [asdict(r) for r in self.results]},
                f, indent=2, default=str,
            )
        print(f"\nSaved {len(self.results)} ablation results to {path}")

    def summary_table(self) -> str:
        """Format as markdown table."""
        lines = [
            "| Experiment | Axis | Variant | F1-macro | Acc | Params (M) |\n"
            "|-----------|------|---------|----------|-----|-----------|"
        ]
        for r in self.results:
            params = f"{r.total_params / 1e6:.2f}"
            lines.append(
                f"| {r.experiment_id} | {r.axis} | {r.variant} | "
                f"{r.f1_macro:.4f} | {r.accuracy:.4f} | {params} |"
            )
        return "\n".join(lines)
