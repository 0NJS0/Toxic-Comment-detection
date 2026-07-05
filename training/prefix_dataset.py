"""
Prefix Dataset for Early Toxicity Detection
=============================================

Wraps a tokenized HuggingFace Dataset and yields prefix-length slices
during training. The model learns to predict toxicity from partial
text prefixes as well as from the full sequence.

This is the key innovation for *while-typing* detection: the model
sees prefixes of lengths [8, 16, 32, 64, 128] tokens and must predict
the SAME labels as the full comment. This trains it to recognize
toxicity patterns early.

Curriculum learning:
    Early epochs: train on longer prefixes (easier — more signal)
    Later epochs: mix in shorter prefixes (harder — less signal)
"""

import random
import numpy as np
from typing import List, Optional


class PrefixDataset:
    """
    Wraps a tokenized dataset and returns prefix-length slices.

    Each sample is a dict with:
        - input_ids: list of ints (variable length, truncated to prefix_len)
        - attention_mask: list of ints (same length as input_ids)
        - labels: list of floats (6 binary labels, same as full text)
        - prefix_length: int (the prefix length used for this sample)
        - full_length: int (original full sequence length)
    """

    def __init__(
        self,
        base_dataset,
        prefix_lengths: List[Optional[int]] = None,
        curriculum_epochs: int = 5,
    ):
        """
        Parameters
        ----------
        base_dataset : Dataset
            The tokenized HF Dataset (from Module 2).
        prefix_lengths : list of int or None
            Token-based prefix lengths to sample from.
            None = full sequence.
            Default: [8, 16, 32, 64, 128]
        curriculum_epochs : int
            Number of epochs over which to phase in short prefixes.
            Epoch 0: only longest prefixes
            Epoch curriculum_epochs-1: all prefixes
        """
        self.base = base_dataset
        self.lengths = prefix_lengths or [8, 16, 32, 64, 128]
        self.curriculum_epochs = curriculum_epochs
        self._epoch = 0

    def set_epoch(self, epoch: int):
        """Set current curriculum epoch."""
        self._epoch = min(epoch, self.curriculum_epochs - 1)

    @property
    def active_lengths(self) -> List[int]:
        """
        Return the subset of prefix lengths active for the current epoch.

        Curriculum schedule:
            epoch 0: only [128] (easiest — closest to full text)
            epoch 1: [64, 128]
            epoch 2: [32, 64, 128]
            epoch 3: [16, 32, 64, 128]
            epoch 4+: [8, 16, 32, 64, 128] (all lengths)
        """
        num_active = min(len(self.lengths), self._epoch + 1)
        return self.lengths[-num_active:]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        full_len = len(item["input_ids"])
        prefix_len = random.choice(self.active_lengths)

        # If prefix is longer than actual sequence, use full sequence
        if prefix_len >= full_len:
            return {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"],
                "id": item.get("id", str(idx)),
                "prefix_length": full_len,
                "full_length": full_len,
            }

        return {
            "input_ids": item["input_ids"][:prefix_len],
            "attention_mask": item["attention_mask"][:prefix_len],
            "labels": item["labels"],
            "id": item.get("id", str(idx)),
            "prefix_length": prefix_len,
            "full_length": full_len,
        }


class PrefixCollator:
    """
    Collator for PrefixDataset.

    Pads variable-length prefix sequences to the longest in the batch,
    just like MultiLabelDataCollator but handles the extra metadata
    (prefix_length, full_length) added by PrefixDataset.
    """

    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding
        self.pad_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        )

    def __call__(self, batch: list) -> dict:
        padded = self.pad_collator(
            [{"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]}
             for s in batch]
        )
        import torch
        labels = torch.tensor([s["labels"] for s in batch], dtype=torch.float32)
        result = {**padded, "labels": labels}
        # Pass through metadata
        if "prefix_length" in batch[0]:
            result["prefix_length"] = [s["prefix_length"] for s in batch]
        if "full_length" in batch[0]:
            result["full_length"] = [s["full_length"] for s in batch]
        return result
