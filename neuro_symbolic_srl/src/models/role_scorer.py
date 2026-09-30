"""Pairwise MLP role-emission head.

Implements Section 2.4 of context.txt: for every token ``i`` and the
predicate ``p``, forms the pairwise representation

    [ h_i ; h_p ; (h_i - h_p) ; h_i ⊙ h_p ]

and projects it through a 2-layer MLP with GELU activation to unconstrained
emission potentials ``psi_neural(i, r)`` over the BIO role label set ``R``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pool_predicate_hidden_state(
    hidden_states: torch.Tensor, predicate_mask: torch.Tensor
) -> torch.Tensor:
    """Average the hidden states at the predicate's (possibly multi-subword) position.

    Args:
        hidden_states: ``(batch, seq_len, hidden_dim)``.
        predicate_mask: ``(batch, seq_len)`` with 1s marking the predicate's
            subword span (at least one 1 per row).

    Returns:
        ``(batch, hidden_dim)`` — h_p per example.
    """
    mask = predicate_mask.float().unsqueeze(-1)  # (B, N, 1)
    denom = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return (hidden_states * mask).sum(dim=1) / denom


class RoleScorer(nn.Module):
    """Pairwise MLP emission scorer producing ``(batch, seq_len, num_labels)`` logits."""

    def __init__(
        self, hidden_dim: int, mlp_hidden_dim: int, num_labels: int, dropout: float
    ) -> None:
        super().__init__()
        pairwise_dim = 4 * hidden_dim
        self.fc1 = nn.Linear(pairwise_dim, mlp_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_hidden_dim, num_labels)

    def forward(
        self, hidden_states: torch.Tensor, predicate_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)`` — h_i for every token.
            predicate_mask: ``(batch, seq_len)`` — predicate indicator mask.

        Returns:
            ``(batch, seq_len, num_labels)`` emission potentials ``psi_neural``.
        """
        predicate_hidden = pool_predicate_hidden_state(hidden_states, predicate_mask)
        predicate_hidden = predicate_hidden.unsqueeze(1).expand_as(hidden_states)

        pairwise = torch.cat(
            [
                hidden_states,
                predicate_hidden,
                hidden_states - predicate_hidden,
                hidden_states * predicate_hidden,
            ],
            dim=-1,
        )
        x = self.dropout(self.activation(self.fc1(pairwise)))
        return self.fc2(x)
