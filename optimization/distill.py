"""
Knowledge Distillation for FedPref
====================================

DistilBERT -> TinyBERT or even smaller student models.

Standard KD: student trained on both:
  - Hard labels (ground truth)
  - Soft targets from teacher (with temperature scaling)

Given our multi-label setup, we use:
  - BCEWithLogitsLoss for hard labels
  - KL divergence (or MSE) between teacher and student logits for soft targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from typing import Optional, Dict


def kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    KL divergence loss between teacher and student logits.

    Higher temperature = softer probability distribution,
    exposing more information about relative label probabilities.
    """
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature ** 2)


class DistillationTrainer:
    """
    Trains a student model using knowledge distillation from a teacher.

    Combined loss:
        L = alpha * BCE(student_logits, hard_labels)
          + (1 - alpha) * temperature^2 * KL(student_logits, teacher_logits)

    For multi-label toxicity, we apply KL per-label and average.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        temperature: float = 4.0,
        alpha: float = 0.5,
        learning_rate: float = 2e-5,
        num_labels: int = 6,
    ):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.num_labels = num_labels

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler=None,
    ) -> Dict[str, float]:
        """Train one epoch. Returns dict of losses."""
        self.student.train()
        total_bce = 0.0
        total_kl = 0.0
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_out.logits if hasattr(teacher_out, "logits") else teacher_out

            # Student forward
            student_out = self.student(input_ids, attention_mask=attention_mask)
            student_logits = student_out.logits if hasattr(student_out, "logits") else student_out

            # BCE loss (hard labels)
            loss_bce = self.bce_loss(student_logits, labels)

            # KL loss (soft targets)
            loss_kl = kl_div_loss(student_logits, teacher_logits, self.temperature)

            # Combined
            loss = self.alpha * loss_bce + (1 - self.alpha) * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.student.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_bce += loss_bce.item()
            total_kl += loss_kl.item()
            total_loss += loss.item()
            num_batches += 1

        return {
            "bce_loss": total_bce / num_batches,
            "kl_loss": total_kl / num_batches,
            "total_loss": total_loss / num_batches,
        }

    def train(
        self,
        train_dataset,
        val_dataset,
        collator,
        num_epochs: int = 5,
        batch_size: int = 16,
        output_dir: str = "results/distillation",
    ) -> Dict:
        """Full distillation training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        history = {"train": [], "val": []}
        best_f1 = 0.0

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, scheduler)
            history["train"].append(train_metrics)

            val_metrics = self.evaluate(val_loader)
            history["val"].append(val_metrics)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val F1: {val_metrics.get('f1_macro', 0):.4f}"
            )

            if val_metrics.get("f1_macro", 0) > best_f1:
                best_f1 = val_metrics["f1_macro"]
                torch.save(
                    {"epoch": epoch, "model_state_dict": self.student.state_dict(), "best_f1": best_f1},
                    output_path / "best_student.pt",
                )
                print(f"  New best student saved (F1: {best_f1:.4f})")

        return history

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate student on validation set."""
        from evaluation.metrics import compute_metrics

        self.student.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                out = self.student(input_ids, attention_mask=attention_mask)
                logits = out.logits if hasattr(out, "logits") else out

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return compute_metrics(all_labels, all_logits)


def distill_knowledge(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_dataset,
    val_dataset,
    collator,
    device: torch.device,
    num_epochs: int = 5,
    temperature: float = 4.0,
    alpha: float = 0.5,
    output_dir: str = "results/distillation",
) -> Dict:
    """Convenience function to run distillation."""
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        device=device,
        temperature=temperature,
        alpha=alpha,
    )
    return trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )
