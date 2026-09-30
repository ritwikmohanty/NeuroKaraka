"""Contextual sequence encoder: Indic transformer + morphology -> 3-layer BiLSTM.

Implements Section 2.1/2.2 of context.txt: each subword token's composite
vector ``e_i = [w_i ; m_i ; p_i]`` (backbone hidden state, morphological
feature embeddings, predicate-indicator embedding) is fed through a 3-layer
Bidirectional LSTM to produce ``h_i``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"
)


def load_model_config(path: Path = _MODEL_CONFIG_PATH) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class SRLEncoder(nn.Module):
    """Backbone + morphology/predicate embeddings -> 3-layer BiLSTM.

    Output ``h`` has shape ``(batch, seq_len, 2 * bilstm.hidden_dim)``.
    """

    def __init__(self, config: Dict | None = None) -> None:
        super().__init__()
        config = config or load_model_config()

        encoder_cfg = config["encoder"]
        morph_cfg = config["morphology"]
        pred_cfg = config["predicate_indicator"]
        bilstm_cfg = config["bilstm"]

        self.backbone = AutoModel.from_pretrained(encoder_cfg["backbone_name"])
        if encoder_cfg.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.case_embedding = nn.Embedding(
            morph_cfg["num_case_markers"],
            morph_cfg["morph_embedding_dim"],
            padding_idx=0,
        )
        self.clitic_embedding = nn.Embedding(
            morph_cfg["num_clitic_tags"],
            morph_cfg["clitic_embedding_dim"],
            padding_idx=0,
        )
        self.predicate_embedding = nn.Embedding(2, pred_cfg["embedding_dim"])

        concat_dim = (
            encoder_cfg["backbone_hidden_dim"]
            + morph_cfg["morph_embedding_dim"]
            + morph_cfg["clitic_embedding_dim"]
            + pred_cfg["embedding_dim"]
        )

        self.bilstm = nn.LSTM(
            input_size=concat_dim,
            hidden_size=bilstm_cfg["hidden_dim"],
            num_layers=bilstm_cfg["num_layers"],
            dropout=bilstm_cfg["dropout"],
            bidirectional=bilstm_cfg["bidirectional"],
            batch_first=True,
        )
        self.output_dim = bilstm_cfg["hidden_dim"] * (
            2 if bilstm_cfg["bidirectional"] else 1
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        case_marker_ids: torch.Tensor,
        clitic_ids: torch.Tensor,
        predicate_mask: torch.Tensor,
    ) -> torch.Tensor:
        backbone_out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        composite = torch.cat(
            [
                backbone_out,
                self.case_embedding(case_marker_ids),
                self.clitic_embedding(clitic_ids),
                self.predicate_embedding(predicate_mask),
            ],
            dim=-1,
        )

        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            composite, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        hidden_states, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=composite.size(1)
        )
        return hidden_states
