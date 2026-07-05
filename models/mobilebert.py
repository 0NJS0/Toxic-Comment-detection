"""
MobileBERT Model for Multi-Label Toxicity Classification
==========================================================

MobileBERT (25M params) is optimized for mobile/edge deployment with
inverted bottleneck structures and optimized attention. It's ~2.6×
smaller than DistilBERT (66M) while retaining higher accuracy than TinyBERT.

Architecture:
    Input (batch, seq_len) → MobileBERT → [CLS] (batch, 512)
    → Dropout → Linear(512→6) → Logits (batch, 6)

Reference:
    https://huggingface.co/google/mobilebert-uncased
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional


class MobileBERTForMultiLabelClassification(nn.Module):
    """
    MobileBERT with a multi-label classification head for toxicity detection.

    Follows the same interface as DistilBERTForMultiLabelClassification
    so that training, evaluation, and LoRA injection all work identically.
    """

    def __init__(
        self,
        model_name: str = "google/mobilebert-uncased",
        num_labels: int = 6,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        self.mobilebert = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

        hidden_size = self.mobilebert.config.hidden_size  # 512
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
        outputs = self.mobilebert(
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
        self.mobilebert.save_pretrained(save_path)
        torch.save(self.classifier.state_dict(), f"{save_path}/classifier.pt")
        torch.save(
            {"dropout": self.dropout.p, "num_labels": self.num_labels},
            f"{save_path}/head_config.pt",
        )

    @classmethod
    def from_pretrained(cls, load_path: str, model_name: str = None):
        head_config = torch.load(f"{load_path}/head_config.pt", map_location="cpu")
        model = cls(
            model_name=model_name or "google/mobilebert-uncased",
            num_labels=head_config["num_labels"],
            dropout=head_config["dropout"],
        )
        model.mobilebert = model.mobilebert.from_pretrained(load_path)
        model.classifier.load_state_dict(
            torch.load(f"{load_path}/classifier.pt", map_location="cpu")
        )
        return model


def create_model(config: dict) -> MobileBERTForMultiLabelClassification:
    model_config = config["model"]
    return MobileBERTForMultiLabelClassification(
        model_name=model_config["name"],
        num_labels=model_config["num_labels"],
        dropout=model_config["dropout"],
        cache_dir=model_config.get("cache_dir"),
    )


if __name__ == "__main__":
    model = MobileBERTForMultiLabelClassification()
    print(f"Model: {model.model_name}")
    print(f"Parameters: {model.get_num_parameters():,}")

    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids, attention_mask)
    print(f"Logits shape: {logits.shape}")
