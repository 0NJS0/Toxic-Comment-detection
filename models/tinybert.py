"""
TinyBERT Model for Multi-Label Toxicity Classification
========================================================

TinyBERT (14M params, 4 layers, 312 hidden) is ~4.7× smaller than
DistilBERT (66M params). It's designed for fast inference on edge
devices while retaining competitive accuracy.

Architecture:
    Input (batch, seq_len) → TinyBERT → [CLS] (batch, 312)
    → Dropout → Linear(312→6) → Logits (batch, 6)

Reference:
    https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class TinyBERTForMultiLabelClassification(nn.Module):
    """
    TinyBERT with a multi-label classification head for toxicity detection.

    Follows the same interface as DistilBERTForMultiLabelClassification
    so that training, evaluation, and LoRA injection all work identically.
    """

    def __init__(
        self,
        model_name: str = "huawei-noah/TinyBERT_General_4L_312D",
        num_labels: int = 6,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        self.tinybert = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        hidden_size = self.tinybert.config.hidden_size  # 312
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_classifier()

    def _init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.tinybert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, save_path: str):
        self.tinybert.save_pretrained(save_path)
        torch.save(self.classifier.state_dict(), f"{save_path}/classifier.pt")
        torch.save(
            {"dropout": self.dropout.p, "num_labels": self.num_labels},
            f"{save_path}/head_config.pt",
        )

    @classmethod
    def from_pretrained(cls, load_path: str, model_name: str = None):
        head_config = torch.load(f"{load_path}/head_config.pt", map_location="cpu")
        model = cls(
            model_name=model_name or "huawei-noah/TinyBERT_General_4L_312D",
            num_labels=head_config["num_labels"],
            dropout=head_config["dropout"],
        )
        model.tinybert = model.tinybert.from_pretrained(load_path)
        model.classifier.load_state_dict(
            torch.load(f"{load_path}/classifier.pt", map_location="cpu")
        )
        return model


def create_model(config: dict) -> TinyBERTForMultiLabelClassification:
    model_config = config["model"]
    return TinyBERTForMultiLabelClassification(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
        cache_dir=model_config.get("cache_dir"),
    )


if __name__ == "__main__":
    model = TinyBERTForMultiLabelClassification()
    print(f"Model: {model.model_name}")
    print(f"Parameters: {model.get_num_parameters():,}")

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids, attention_mask)
    print(f"Logits shape: {logits.shape}")
