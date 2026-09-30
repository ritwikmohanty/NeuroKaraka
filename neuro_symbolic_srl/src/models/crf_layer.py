"""Custom Linear-Chain CRF with additive ontology-mask support.

Implements Section 2.5 of context.txt:

    Phi = Psi_neural + M
    S(X, y) = sum_i Phi(i, y_i) + sum_i A[y_i, y_{i+1}]
    L_CRF = log Z(X) - S(X, y*)

``Z(X)`` (the log-partition function) is computed via the forward algorithm
in log-space using ``torch.logsumexp``, which stays numerically stable when
``M`` drives masked entries of ``Phi`` to a large finite negative sentinel
(``mask_negative_value: -10000.0`` in ``config/model_config.yaml``, which is
what every caller in this project uses) — logsumexp simply assigns those
paths ~0 probability mass without over/underflowing into NaN.

Note: a literal ``-inf`` mask is *not* supported and can produce NaN
gradients — if an entire logsumexp reduction over some dimension has every
entry equal to ``-inf``, ``torch.logsumexp``'s internal max-subtraction
computes ``-inf - (-inf) = NaN`` in the backward pass. This is the standard
reason production CRF/HMM implementations mask with a large finite negative
number rather than true ``-inf``; do the same here.

Contract: ``mask`` must be a standard **prefix** mask per sequence (1s for a
leading run of real timesteps, then 0s for right-padding only — exactly what
the dataset's ``attention_mask`` already is). ``tags`` must contain valid
label indices in ``[0, num_labels)`` at every position, including padded
ones (their value is irrelevant there since the mask excludes their
contribution, but they must not be -100/out-of-range or torch indexing will
error) — callers should map ignore-index (-100) positions to any valid
placeholder id (e.g. the "O" id) before calling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """A linear-chain CRF whose emissions are ``Phi = Psi_neural + M``."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.start_transitions = nn.Parameter(torch.empty(num_labels).uniform_(-0.1, 0.1))
        self.end_transitions = nn.Parameter(torch.empty(num_labels).uniform_(-0.1, 0.1))
        self.transitions = nn.Parameter(
            torch.empty(num_labels, num_labels).uniform_(-0.1, 0.1)
        )

    @staticmethod
    def combine_emissions(
        neural_emissions: torch.Tensor, ontology_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Phi = Psi_neural + M (identity if no mask is supplied)."""
        if ontology_mask is None:
            return neural_emissions
        return neural_emissions + ontology_mask

    def log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Z(X) in log-space via the forward algorithm. Returns ``(batch,)``."""
        batch_size, seq_len, num_labels = emissions.shape
        mask = mask.bool()

        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)

        for t in range(1, seq_len):
            # (B, C_from, 1) + (1, C_from, C_to) + (B, 1, C_to) -> (B, C_from, C_to)
            broadcast_alpha = alpha.unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)
            broadcast_emit = emissions[:, t].unsqueeze(1)
            scores = broadcast_alpha + broadcast_trans + broadcast_emit
            new_alpha = torch.logsumexp(scores, dim=1)  # (B, C_to)

            mask_t = mask[:, t].unsqueeze(1)  # (B, 1)
            alpha = torch.where(mask_t, new_alpha, alpha)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def gold_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """S(X, y*) for the gold tag sequence. Returns ``(batch,)``."""
        batch_size, seq_len, _ = emissions.shape
        mask_f = mask.float()
        batch_idx = torch.arange(batch_size, device=emissions.device)

        score = self.start_transitions[tags[:, 0]] + emissions[batch_idx, 0, tags[:, 0]]

        for t in range(1, seq_len):
            emit_score = emissions[batch_idx, t, tags[:, t]]
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + mask_f[:, t] * (trans_score + emit_score)

        last_valid_index = mask.long().sum(dim=1) - 1  # (B,)
        last_tags = tags[batch_idx, last_valid_index]
        score = score + self.end_transitions[last_tags]
        return score

    def forward(
        self,
        neural_emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        ontology_mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Negative log-likelihood loss: log Z(X) - S(X, y*)."""
        phi = self.combine_emissions(neural_emissions, ontology_mask)
        log_z = self.log_partition(phi, mask)
        gold = self.gold_score(phi, tags, mask)
        nll = log_z - gold

        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        if reduction == "none":
            return nll
        raise ValueError(f"Unknown reduction: {reduction!r}")
