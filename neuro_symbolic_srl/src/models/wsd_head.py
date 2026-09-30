"""Predicate Word Sense Disambiguation head: dot-product attention over IndoWordNet glosses.

Implements Section 2.3 of context.txt:

    alpha_k = softmax_k( h_p^T W_sense g_k )

``W_sense`` projects the predicate's BiLSTM hidden state ``h_p`` into the
gloss-embedding space, where it is dot-producted against each candidate
synset's gloss embedding ``g_k`` to form attention logits over senses.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WSDHead(nn.Module):
    """Scores candidate IndoWordNet synsets for a predicate's hidden state.

    ``candidate_mask`` handles the variable, per-example number of candidate
    senses ``K`` (padded to a common ``K_max`` in a batch): masked-out
    candidates get ``-inf`` logits so they receive zero attention weight.
    """

    def __init__(self, hidden_dim: int, gloss_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, projection_dim)
        self.gloss_proj = nn.Linear(gloss_dim, projection_dim)

    def forward(
        self,
        predicate_hidden: torch.Tensor,
        gloss_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            predicate_hidden: ``(batch, hidden_dim)`` — h_p per example.
            gloss_embeddings: ``(batch, K_max, gloss_dim)`` — candidate gloss vectors.
            candidate_mask: ``(batch, K_max)`` — 1 for a real candidate, 0 for padding.

        Returns:
            ``(alpha, logits)`` each of shape ``(batch, K_max)``: the softmax
            sense-attention distribution and its pre-softmax logits (the
            latter usable directly as cross-entropy logits for the WSD loss).
        """
        query = self.query_proj(predicate_hidden).unsqueeze(1)  # (B, 1, P)
        keys = self.gloss_proj(gloss_embeddings)  # (B, K, P)

        logits = torch.matmul(keys, query.transpose(1, 2)).squeeze(-1)  # (B, K)
        logits = logits.masked_fill(candidate_mask == 0, float("-inf"))

        alpha = torch.softmax(logits, dim=-1)
        return alpha, logits
