"""
FedPref Federated Learning Server
===================================

Implements the federated averaging loop directly (without Flower's
simulation API) for full control over the aggregation process.

Loop:
  1. Initialize global model
  2. For each round:
     a. Send global params to each client
     b. Clients train locally (LoRA + classifier only)
     c. Aggregate: global = mean(client_params)
     d. Evaluate on global validation set
  3. Save checkpoints + history

This approach gives exact reproducibility and is simpler to debug
than Flower's Ray-based simulation.
"""

import time
import json
import copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.registry import create_model
from models.lora import get_lora_params, get_lora_weights, set_lora_weights
from evaluation.metrics import compute_metrics
from federated.client import FedPrefClient, get_parameters_ndarrays, set_parameters


def run_federated_round(
    clients: List[FedPrefClient],
    global_params: List,
    round_idx: int,
    config: dict,
) -> tuple:
    """
    Execute one federated round: each client trains locally, return updates.

    Returns
    -------
    tuple: (client_updates, client_metrics)
    """
    client_updates = []
    client_metrics = []

    for client in clients:
        # Client training
        params, num_examples, metrics = client.fit(global_params, {
            "local_epochs": config["federated"].get("local_epochs", 1),
            "round": round_idx,
        })
        client_updates.append(params)
        client_metrics.append(metrics)

    return client_updates, client_metrics


def fedavg_aggregate(client_updates: List[List]) -> List:
    """Federated Averaging: compute the mean of all client parameter updates."""
    num_clients = len(client_updates)
    num_params = len(client_updates[0])

    aggregated = []
    for param_idx in range(num_params):
        param_sum = None
        for client_idx in range(num_clients):
            param = np.array(client_updates[client_idx][param_idx], dtype=np.float32)
            if param_sum is None:
                param_sum = param
            else:
                param_sum += param
        aggregated.append(param_sum / num_clients)

    return aggregated


def evaluate_global_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict:
    """Evaluate model on a dataloader and return metrics."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = avg_loss
    return metrics


def run_federated(
    config: dict,
    clients: List[FedPrefClient],
    global_val_dataset,
    tokenizer,
    device: torch.device,
    output_dir: str = "results/federated",
) -> Dict:
    """
    Execute the full federated learning loop.

    Parameters
    ----------
    config : dict
        Full project configuration.
    clients : list of FedPrefClient
        One client per partition.
    global_val_dataset : Dataset
        Global held-out validation set (not seen by any client).
    tokenizer : AutoTokenizer
        For data collation.
    device : torch.device
    output_dir : str
        Where to save checkpoints and logs.

    Returns
    -------
    dict
        Full training history and final metrics.
    """
    fed_config = config["federated"]
    num_clients = len(clients)
    num_rounds = fed_config["num_rounds"]
    strategy_name = fed_config.get("strategy", "fedavg")
    personalize = fed_config.get("personalize", "lora")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Setup
    from training.train import MultiLabelDataCollator
    data_collator = MultiLabelDataCollator(tokenizer)
    val_loader = DataLoader(
        global_val_dataset,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        collate_fn=data_collator,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # Initialize global model
    global_model = create_model(config).to(device)
    global_params = get_parameters_ndarrays(global_model)

    print(f"\n{'='*60}")
    print(f"FEDPREF FEDERATED LEARNING")
    print(f"{'='*60}")
    print(f"  Strategy:    {strategy_name}")
    print(f"  Clients:     {num_clients}")
    print(f"  Rounds:      {num_rounds}")
    print(f"  Personalize: {personalize}")
    print(f"  Params/client: {sum(p.size for p in global_params):,}")
    print(f"{'='*60}")

    # Training loop
    history = {
        "round": [],
        "global_val_loss": [],
        "global_val_accuracy": [],
        "global_val_f1_macro": [],
        "client_losses": [],
        "time_per_round": [],
    }

    best_f1 = 0.0
    start_time = time.time()

    for round_idx in range(1, num_rounds + 1):
        round_start = time.time()

        # Step 1: Clients train locally
        client_updates, client_metrics = run_federated_round(
            clients, global_params, round_idx, config,
        )

        # Step 2: Aggregate (FedAvg)
        global_params = fedavg_aggregate(client_updates)

        # Step 3: Evaluate global model
        set_parameters(global_model, global_params)
        val_metrics = evaluate_global_model(
            global_model, val_loader, loss_fn, device,
        )

        round_time = time.time() - round_start

        # Log
        avg_client_loss = np.mean([m.get("loss", 0) for m in client_metrics])
        history["round"].append(round_idx)
        history["global_val_loss"].append(val_metrics["loss"])
        history["global_val_accuracy"].append(val_metrics["accuracy"])
        history["global_val_f1_macro"].append(val_metrics["f1_macro"])
        history["client_losses"].append(float(avg_client_loss))
        history["time_per_round"].append(round_time)

        print(f"\n  Round {round_idx}/{num_rounds} "
              f"({round_time:.1f}s)")
        print(f"    Client loss (avg):  {avg_client_loss:.4f}")
        print(f"    Global val loss:    {val_metrics['loss']:.4f}")
        print(f"    Global val acc:     {val_metrics['accuracy']:.4f}")
        print(f"    Global val F1:      {val_metrics['f1_macro']:.4f}")

        # Save best model
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_path = out_path / "best_model"
            best_path.mkdir(exist_ok=True)
            global_model.save_pretrained(str(best_path))
            print(f"    ✓ New best model saved (F1={best_f1:.4f})")

        # Save round checkpoint
        if round_idx % 5 == 0 or round_idx == num_rounds:
            ckpt = {
                "round": round_idx,
                "global_params": global_params,
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(ckpt, out_path / f"checkpoint_round_{round_idx}.pt")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Best global F1-macro: {best_f1:.4f}")
    print(f"{'='*60}")

    # Final evaluation
    print(f"\n[Final] Evaluating global model on all clients...")
    per_client_metrics = []
    for cid, client in enumerate(clients):
        _, _, eval_metrics = client.evaluate(global_params, {})
        per_client_metrics.append(eval_metrics)
        print(f"  Client {cid}: F1={eval_metrics.get('f1_macro', 0):.4f}")

    client_f1s = [m.get("f1_macro", 0) for m in per_client_metrics]
    print(f"\n  Mean client F1: {np.mean(client_f1s):.4f} ± {np.std(client_f1s):.4f}")

    # Save results
    results = {
        "strategy": strategy_name,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "best_global_f1_macro": float(best_f1),
        "mean_client_f1_macro": float(np.mean(client_f1s)),
        "std_client_f1_macro": float(np.std(client_f1s)),
        "history": history,
        "per_client_metrics": {
            f"client_{i}": m for i, m in enumerate(per_client_metrics)
        },
    }

    history_path = out_path / "federated_history.json"
    with open(history_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {history_path}")

    return results
