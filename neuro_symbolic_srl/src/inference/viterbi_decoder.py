"""Constrained Viterbi decoding over masked potentials Phi, plus uniqueness post-processing.

Implements Section 2.5 / Phase 5 of context.txt: dynamic-programming search
for the best label sequence under ``Phi = Psi_neural + M`` and the learned
transition matrix ``A``, followed by a post-processing pass that enforces
the structural invariant that a monovalent/bivalent verbal frame admits at
most one span of each core argument role (ARG0, ARG1): a linear-chain CRF's
Markov transitions cannot express that *global* constraint on their own, so
if decoding still yields more than one span of a constrained role, the
highest-scoring span is kept and the rest are demoted to ``O``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def viterbi_decode(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[List[int], float]:
    """Best label-index path for a single sequence under Phi + A.

    Args:
        emissions: ``(seq_len, num_labels)`` — Phi = Psi_neural + M.
        transitions: ``(num_labels, num_labels)`` — A[from, to].
        start_transitions, end_transitions: ``(num_labels,)`` boundary scores.
        mask: optional ``(seq_len,)`` boolean prefix mask (see CRF contract);
            only the valid leading prefix is decoded.

    Returns:
        ``(label_indices, path_score)``.
    """
    if mask is not None:
        valid_len = int(mask.bool().sum().item())
        emissions = emissions[:valid_len]

    seq_len, num_labels = emissions.shape
    if seq_len == 0:
        return [], 0.0

    backpointers = torch.zeros(seq_len, num_labels, dtype=torch.long)
    score = start_transitions + emissions[0]  # (C,)

    for t in range(1, seq_len):
        # (C_from, 1) + (C_from, C_to) -> (C_from, C_to)
        broadcast_score = score.unsqueeze(1) + transitions
        best_prev_score, best_prev_label = broadcast_score.max(dim=0)  # over C_from
        score = best_prev_score + emissions[t]
        backpointers[t] = best_prev_label

    score = score + end_transitions
    best_last_label = int(torch.argmax(score).item())
    best_score = float(score[best_last_label].item())

    path = [best_last_label]
    for t in range(seq_len - 1, 0, -1):
        best_last_label = int(backpointers[t, best_last_label].item())
        path.append(best_last_label)
    path.reverse()

    return path, best_score


def batch_viterbi_decode(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    mask: torch.Tensor,
) -> List[List[int]]:
    """Per-example Viterbi decode over a batch. Returns one variable-length path per row."""
    batch_size = emissions.size(0)
    return [
        viterbi_decode(
            emissions[b], transitions, start_transitions, end_transitions, mask[b]
        )[0]
        for b in range(batch_size)
    ]


def _find_spans(labels: Sequence[str], role: str) -> List[Tuple[int, int]]:
    """Contiguous (start, end) index ranges (inclusive) of a `B-role (I-role)*` span."""
    spans: List[Tuple[int, int]] = []
    i, n = 0, len(labels)
    b_tag, i_tag = f"B-{role}", f"I-{role}"
    while i < n:
        if labels[i] == b_tag:
            start = i
            j = i + 1
            while j < n and labels[j] == i_tag:
                j += 1
            spans.append((start, j - 1))
            i = j
        else:
            i += 1
    return spans


def enforce_uniqueness(
    labels: List[str],
    emissions: torch.Tensor,
    label_to_index: Dict[str, int],
    roles: Sequence[str] = ("ARG0", "ARG1"),
) -> List[str]:
    """Demote all but the highest-scoring span of each constrained role to O.

    ``emissions`` must be the same ``(seq_len, num_labels)`` Phi used to
    decode ``labels`` (post-mask), so a span's score is the sum of the
    emission potentials it actually received along the winning path.
    """
    labels = list(labels)
    for role in roles:
        spans = _find_spans(labels, role)
        if len(spans) <= 1:
            continue

        span_scores = []
        for start, end in spans:
            span_score = 0.0
            for position in range(start, end + 1):
                label_index = label_to_index[labels[position]]
                span_score += float(emissions[position, label_index].item())
            span_scores.append(span_score)

        best_span_index = max(range(len(spans)), key=lambda i: span_scores[i])
        for i, (start, end) in enumerate(spans):
            if i == best_span_index:
                continue
            for position in range(start, end + 1):
                labels[position] = "O"

    return labels


def subword_labels_to_word_labels(
    subword_labels: Sequence[str],
    word_ids: Sequence[Optional[int]],
    num_words: int,
) -> List[str]:
    """Collapse subword-level BIO labels to one label per word.

    Takes the label of each word's first subword (where B-/I- boundaries are
    decided); continuation subwords of the same word are redundant by the
    ``_propagate_bio_to_subwords`` convention used when building targets.
    """
    word_labels = ["O"] * num_words
    seen = set()
    for label, word_id in zip(subword_labels, word_ids):
        if word_id is not None and word_id not in seen:
            word_labels[word_id] = label
            seen.add(word_id)
    return word_labels


def decode(
    emissions: torch.Tensor,
    transitions: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    label_list: List[str],
    mask: Optional[torch.Tensor] = None,
    unique_roles: Sequence[str] = ("ARG0", "ARG1"),
) -> List[str]:
    """Full decode pipeline: Viterbi search -> label strings -> uniqueness post-processing."""
    label_indices, _ = viterbi_decode(
        emissions, transitions, start_transitions, end_transitions, mask
    )
    valid_len = len(label_indices)
    labels = [label_list[idx] for idx in label_indices]

    label_to_index = {label: i for i, label in enumerate(label_list)}
    return enforce_uniqueness(labels, emissions[:valid_len], label_to_index, unique_roles)
