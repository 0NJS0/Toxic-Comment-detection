"""
Prefix-Based Training Loop
============================

Trains a DistilBERT model (with or without LoRA) to predict toxicity
from partial text prefixes using curriculum learning.

Key idea: the model sees prefixes of lengths [8, 16, 32, 64, 128] tokens
during training and must predict the same labels as the full comment.
This forces the model to learn *early* toxicity indicators.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict

from training.prefix_dataset import PrefixDataset, PrefixCollator
from models.registry import create_model
from evaluation.metrics import compute_metrics


def train_prefix_model(
    config: dict,
    train_dataset,
    val_dataset,
    tokenizer,
    device: torch.device,
    output_dir: str = "results/prefix",
    resume_from: Optional[str] = None,
) -> Dict:
    """
    Train a model for prefix-based early toxicity detection.

    Parameters
    ----------
    config : dict
        The project config dict.
    train_dataset : Dataset
        Tokenized training dataset (from Module 2).
    val_dataset : Dataset
        Tokenized validation dataset.
    tokenizer : AutoTokenizer
        Pretrained tokenizer (not used in training but passed to collator).
    device : torch.device
        Device to train on.
    output_dir : str
        Path to save checkpoints and metrics.
    resume_from : str or None
        Path to a checkpoint to resume from.

    Returns
    -------
    dict : Training metrics (loss, accuracy, F1, AUROC per epoch).
    """
    train_cfg = config["training"]
    prefix_cfg = config["prefix"]
    model_name = config["model"]["name"]
    label_cols = config["data"]["labels"]
    lora_cfg = config.get("personalization", {})
    batch_size = train_cfg.get("batch_size", 16)
    num_epochs = prefix_cfg.get("num_epochs", 10)
    learning_rate = train_cfg.get("learning_rate", 2e-5)
    curriculum_epochs = prefix_cfg.get("curriculum_epochs", 5)
    prefix_lengths = prefix_cfg.get("prefix_lengths", [8, 16, 32, 64, 128])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create prefix datasets with curriculum
    train_prefix_ds = PrefixDataset(
        train_dataset,
        prefix_lengths=prefix_lengths,
        curriculum_epochs=curriculum_epochs,
    )
    collator = PrefixCollator(tokenizer)

    # Build model
    model = create_model(
        model_name=model_name,
        num_labels=len(label_cols),
        lora_config=lora_cfg if lora_cfg.get("enabled", False) else None,
    )
    model.to(device)

    # Optimizer: only trainable params (LoRA + classifier head)
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)

    # Loss: BCEWithLogitsLoss for multi-label
    criterion = nn.BCEWithLogitsLoss()

    # LR scheduler with warmup
    # Note: num_batches not known until first batch — set later
    scheduler = None

    start_epoch = 0
    best_f1 = 0.0
    history = {"train_loss": [], "val_metrics": []}

    # Resume from checkpoint
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        history = checkpoint.get("history", history)
        print(f"Resumed from epoch {checkpoint['epoch']} (best F1: {best_f1:.4f})")

    for epoch in range(start_epoch, num_epochs):
        train_prefix_ds.set_epoch(epoch)
        train_loader = DataLoader(
            train_prefix_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )

        # Init scheduler on first epoch
        if scheduler is None:
            total_steps = len(train_loader) * (num_epochs - start_epoch)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )

        # Training
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        history["train_loss"].append(avg_loss)

        # Validation on full sequences (no prefix truncation)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator.pad_collator,
            num_workers=0,
        )

        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_metrics(
            all_labels, all_logits, label_cols=label_cols,
        )
        print(
            f"  Val | Accuracy: {metrics['accuracy']:.4f} | "
            f"F1-macro: {metrics['f1_macro']:.4f} | "
            f"AUROC: {metrics['roc_auc']:.4f}"
        )
        history["val_metrics"].append(metrics)

        # Save best model
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "history": history,
                    "config": config,
                },
                output_path / "best_model.pt",
            )
            print(f"  New best model saved (F1: {best_f1:.4f})")

        # Save per-epoch checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "history": history,
                "config": config,
            },
            output_path / f"checkpoint_epoch_{epoch+1}.pt",
        )

        # Save updated metrics
        history["best_f1"] = best_f1
        with open(output_path / "training_metrics.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

    return history


def evaluate_prefix_model(
    model,
    dataset,
    tokenizer,
    device: torch.device,
    label_cols: list,
    prefix_lengths=None,
) -> Dict:
    """
    Evaluate a model at multiple prefix lengths.

    Measures how detection quality degrades as prefix shortens.
    The difference between full-text metrics and prefix-N metrics
    is the *early detection cost*.

    Returns
    -------
    dict of {prefix_length: metrics_dict}
    """
    from evaluation.metrics import compute_metrics
    from training.prefix_dataset import PrefixDataset, PrefixCollator

    prefix_lengths = prefix_lengths or [8, 16, 32, 64, 128, None]
    collator = PrefixCollator(tokenizer)
    results = {}

    model.eval()

    for plen in prefix_lengths:
        prefix_ds = PrefixDataset(dataset, prefix_lengths=[plen] if plen else [])
        prefix_ds.set_epoch(99)
        loader = DataLoader(
            prefix_ds,
            batch_size=16,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        all_logits = []
        all_labels = []
        all_prefix_lens = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_metrics(all_labels, all_logits, label_cols=label_cols)
        key = f"prefix_{plen}" if plen else "full"
        results[key] = {
            "prefix_length_tokens": plen,
            **metrics,
        }

    return results
