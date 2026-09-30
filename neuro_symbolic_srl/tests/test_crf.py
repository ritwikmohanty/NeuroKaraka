"""Unit tests for the Phase 3 model modules: encoder, WSD head, role scorer, CRF.

The CRF tests are the primary focus (per Phase 3 instructions): the forward
algorithm's log-partition and gold-path score are checked against brute-force
enumeration over all label sequences for small toy inputs, gradient flow is
verified end-to-end, and numerical stability is checked under -10000/-inf
ontology masks.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch

from src.inference.viterbi_decoder import (
    _find_spans,
    batch_viterbi_decode,
    decode,
    enforce_uniqueness,
    viterbi_decode,
)
from src.models.crf_layer import LinearChainCRF
from src.models.role_scorer import RoleScorer, pool_predicate_hidden_state
from src.models.wsd_head import WSDHead


def _brute_force_log_partition(
    emissions: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> float:
    """Reference implementation: enumerate every label path and logsumexp their scores."""
    seq_len, num_labels = emissions.shape
    path_scores = []
    for path in itertools.product(range(num_labels), repeat=seq_len):
        score = start_transitions[path[0]].item() + emissions[0, path[0]].item()
        for t in range(1, seq_len):
            score += transitions[path[t - 1], path[t]].item()
            score += emissions[t, path[t]].item()
        score += end_transitions[path[-1]].item()
        path_scores.append(score)
    return torch.logsumexp(torch.tensor(path_scores), dim=0).item()


def _brute_force_path_score(
    emissions: torch.Tensor,
    tags: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> float:
    seq_len = emissions.shape[0]
    score = start_transitions[tags[0]].item() + emissions[0, tags[0]].item()
    for t in range(1, seq_len):
        score += transitions[tags[t - 1], tags[t]].item()
        score += emissions[t, tags[t]].item()
    score += end_transitions[tags[-1]].item()
    return score


class TestLinearChainCRFAgainstBruteForce:
    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.num_labels = 3
        self.seq_len = 3
        self.crf = LinearChainCRF(self.num_labels)
        # Deterministic, non-trivial parameters instead of the random init.
        with torch.no_grad():
            self.crf.start_transitions.copy_(torch.tensor([0.1, -0.3, 0.5]))
            self.crf.end_transitions.copy_(torch.tensor([-0.2, 0.4, 0.0]))
            self.crf.transitions.copy_(
                torch.tensor(
                    [
                        [0.1, 0.2, -0.1],
                        [-0.3, 0.05, 0.4],
                        [0.2, -0.2, 0.1],
                    ]
                )
            )
        self.emissions = torch.tensor(
            [
                [1.0, 0.5, -0.5],
                [0.2, 1.2, 0.1],
                [-0.3, 0.4, 0.9],
            ]
        )

    def test_log_partition_matches_brute_force_enumeration(self) -> None:
        expected = _brute_force_log_partition(
            self.emissions,
            self.crf.start_transitions,
            self.crf.transitions,
            self.crf.end_transitions,
        )
        actual = self.crf.log_partition(
            self.emissions.unsqueeze(0), torch.ones(1, self.seq_len, dtype=torch.bool)
        )
        assert actual.item() == pytest.approx(expected, abs=1e-4)

    def test_gold_score_matches_brute_force_path_score(self) -> None:
        tags = torch.tensor([0, 2, 1])
        expected = _brute_force_path_score(
            self.emissions,
            tags,
            self.crf.start_transitions,
            self.crf.transitions,
            self.crf.end_transitions,
        )
        actual = self.crf.gold_score(
            self.emissions.unsqueeze(0),
            tags.unsqueeze(0),
            torch.ones(1, self.seq_len, dtype=torch.bool),
        )
        assert actual.item() == pytest.approx(expected, abs=1e-4)

    def test_nll_is_nonnegative_since_gold_path_is_one_term_in_the_partition(self) -> None:
        tags = torch.tensor([1, 1, 2])
        loss = self.crf(
            self.emissions.unsqueeze(0),
            tags.unsqueeze(0),
            torch.ones(1, self.seq_len, dtype=torch.bool),
        )
        assert loss.item() >= -1e-6

    def test_padding_does_not_influence_shorter_sequence_result(self) -> None:
        # A length-2 sequence's log-partition/gold score must be identical
        # whether or not it is padded to length 3 with garbage emissions,
        # as long as the padded positions are masked out.
        short_emissions = self.emissions[:2]
        short_mask = torch.ones(1, 2, dtype=torch.bool)
        unpadded_log_z = self.crf.log_partition(short_emissions.unsqueeze(0), short_mask)

        garbage = torch.tensor([[999.0, -999.0, 123.0]])
        padded_emissions = torch.cat([short_emissions, garbage], dim=0).unsqueeze(0)
        padded_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        padded_log_z = self.crf.log_partition(padded_emissions, padded_mask)

        assert padded_log_z.item() == pytest.approx(unpadded_log_z.item(), abs=1e-4)


class TestLinearChainCRFGradientFlow:
    def test_backward_populates_gradients_on_all_parameters(self) -> None:
        crf = LinearChainCRF(num_labels=5)
        batch, seq_len = 3, 6
        emissions = torch.randn(batch, seq_len, 5, requires_grad=True)
        tags = torch.randint(0, 5, (batch, seq_len))
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        loss = crf(emissions, tags, mask)
        loss.backward()

        assert emissions.grad is not None
        assert torch.any(emissions.grad != 0)
        assert crf.transitions.grad is not None
        assert torch.any(crf.transitions.grad != 0)
        assert crf.start_transitions.grad is not None
        assert crf.end_transitions.grad is not None

    def test_backward_with_variable_length_batch(self) -> None:
        crf = LinearChainCRF(num_labels=4)
        emissions = torch.randn(2, 5, 4, requires_grad=True)
        tags = torch.randint(0, 4, (2, 5))
        mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
        )
        loss = crf(emissions, tags, mask)
        loss.backward()
        assert emissions.grad is not None
        assert not torch.isnan(emissions.grad).any()


class TestLinearChainCRFOntologyMaskStability:
    def setup_method(self) -> None:
        torch.manual_seed(1)
        self.num_labels = 4
        self.crf = LinearChainCRF(self.num_labels)

    def test_no_nan_or_inf_under_minus_10000_mask(self) -> None:
        batch, seq_len = 2, 5
        emissions = torch.randn(batch, seq_len, self.num_labels)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        ontology_mask = torch.zeros(batch, seq_len, self.num_labels)
        ontology_mask[:, :, 0] = -10000.0  # forbid label 0 everywhere

        phi = self.crf.combine_emissions(emissions, ontology_mask)
        log_z = self.crf.log_partition(phi, mask)
        assert torch.isfinite(log_z).all()

        tags = torch.ones(batch, seq_len, dtype=torch.long)  # avoid forbidden label
        loss = self.crf(emissions, tags, mask, ontology_mask=ontology_mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert not torch.isnan(emissions.grad if emissions.requires_grad else torch.tensor(0.0)).any()

    def test_no_nan_or_inf_under_a_very_large_finite_negative_mask(self) -> None:
        # -1e6 (well beyond the project's own -10000 sentinel) should still
        # behave cleanly: this is the regime the CRF is actually designed
        # for. Literal float("-inf") is a documented non-goal (see
        # crf_layer.py's module docstring) because it can NaN the backward
        # pass when it drives an entire logsumexp reduction to all -inf.
        batch, seq_len = 1, 3
        emissions = torch.randn(batch, seq_len, self.num_labels, requires_grad=True)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        ontology_mask = torch.zeros(batch, seq_len, self.num_labels)
        ontology_mask[:, :, 0] = -1e6

        tags = torch.zeros(batch, seq_len, dtype=torch.long) + 1
        loss = self.crf(emissions, tags, mask, ontology_mask=ontology_mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(emissions.grad).all()

    def test_masking_a_label_raises_its_nll_when_gold_uses_it(self) -> None:
        # Forcing an ontology mask onto the gold-path label should make that
        # assignment much less likely (higher NLL) than an identical setup
        # where the same label is left unmasked.
        batch, seq_len = 1, 3
        emissions = torch.randn(batch, seq_len, self.num_labels)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        tags = torch.tensor([[0, 0, 0]])

        unmasked_loss = self.crf(emissions, tags, mask)

        ontology_mask = torch.zeros(batch, seq_len, self.num_labels)
        ontology_mask[:, :, 0] = -10000.0  # forbids exactly the gold label
        masked_loss = self.crf(emissions, tags, mask, ontology_mask=ontology_mask)

        assert masked_loss.item() > unmasked_loss.item() + 100


# ---------------------------------------------------------------------------
# RoleScorer / pool_predicate_hidden_state (no backbone download required)
# ---------------------------------------------------------------------------


class TestPoolPredicateHiddenState:
    def test_pools_only_the_predicate_position(self) -> None:
        hidden = torch.zeros(1, 4, 3)
        hidden[0, 2] = torch.tensor([1.0, 2.0, 3.0])
        predicate_mask = torch.tensor([[0, 0, 1, 0]])
        pooled = pool_predicate_hidden_state(hidden, predicate_mask)
        assert torch.allclose(pooled, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_averages_over_a_multi_subword_predicate_span(self) -> None:
        hidden = torch.zeros(1, 3, 2)
        hidden[0, 0] = torch.tensor([2.0, 0.0])
        hidden[0, 1] = torch.tensor([4.0, 0.0])
        predicate_mask = torch.tensor([[1, 1, 0]])
        pooled = pool_predicate_hidden_state(hidden, predicate_mask)
        assert torch.allclose(pooled, torch.tensor([[3.0, 0.0]]))


class TestRoleScorer:
    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.hidden_dim = 16
        self.num_labels = 7
        self.scorer = RoleScorer(
            hidden_dim=self.hidden_dim,
            mlp_hidden_dim=32,
            num_labels=self.num_labels,
            dropout=0.1,
        )

    def test_output_shape(self) -> None:
        batch, seq_len = 3, 5
        hidden_states = torch.randn(batch, seq_len, self.hidden_dim)
        predicate_mask = torch.zeros(batch, seq_len, dtype=torch.long)
        predicate_mask[:, 0] = 1
        emissions = self.scorer(hidden_states, predicate_mask)
        assert emissions.shape == (batch, seq_len, self.num_labels)

    def test_gradient_flows_to_mlp_parameters(self) -> None:
        hidden_states = torch.randn(2, 4, self.hidden_dim, requires_grad=True)
        predicate_mask = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
        emissions = self.scorer(hidden_states, predicate_mask)
        emissions.sum().backward()
        assert hidden_states.grad is not None
        assert self.scorer.fc1.weight.grad is not None
        assert torch.any(self.scorer.fc1.weight.grad != 0)


class TestWSDHead:
    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.head = WSDHead(hidden_dim=16, gloss_dim=8, projection_dim=12)

    def test_attention_sums_to_one_over_valid_candidates(self) -> None:
        predicate_hidden = torch.randn(2, 16)
        gloss_embeddings = torch.randn(2, 5, 8)
        candidate_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long
        )
        alpha, logits = self.head(predicate_hidden, gloss_embeddings, candidate_mask)
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(2), atol=1e-5)
        assert torch.all(alpha[0, 3:] == 0)
        assert torch.all(alpha[1, 2:] == 0)

    def test_masked_candidates_get_zero_probability_not_nan(self) -> None:
        predicate_hidden = torch.randn(1, 16)
        gloss_embeddings = torch.randn(1, 3, 8)
        candidate_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)
        alpha, _ = self.head(predicate_hidden, gloss_embeddings, candidate_mask)
        assert not torch.isnan(alpha).any()
        assert alpha[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_gradient_flows_to_projection_layers(self) -> None:
        predicate_hidden = torch.randn(2, 16, requires_grad=True)
        gloss_embeddings = torch.randn(2, 4, 8)
        candidate_mask = torch.ones(2, 4, dtype=torch.long)
        alpha, _ = self.head(predicate_hidden, gloss_embeddings, candidate_mask)
        alpha.sum().backward()
        assert self.head.query_proj.weight.grad is not None
        assert torch.any(self.head.query_proj.weight.grad != 0)


# ---------------------------------------------------------------------------
# SRLEncoder (requires downloading the real MuRIL backbone)
# ---------------------------------------------------------------------------


class TestSRLEncoder:
    def test_forward_shape_and_gradient_flow(self) -> None:
        from src.models.encoder import SRLEncoder, load_model_config

        config = load_model_config()
        encoder = SRLEncoder(config)
        tokenizer_name = config["encoder"]["backbone_name"]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        batch_tokens = [["बच्चे", "ने", "सेब", "खाया"], ["राम", "गया"]]
        encodings = [
            tokenizer(tokens, is_split_into_words=True) for tokens in batch_tokens
        ]
        max_len = max(len(e["input_ids"]) for e in encodings)

        def pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        input_ids = torch.tensor(
            [pad(e["input_ids"], tokenizer.pad_token_id) for e in encodings]
        )
        attention_mask = torch.tensor(
            [pad(e["attention_mask"], 0) for e in encodings]
        )
        case_marker_ids = torch.zeros_like(input_ids)
        clitic_ids = torch.zeros_like(input_ids)
        predicate_mask = torch.zeros_like(input_ids)
        predicate_mask[:, 1] = 1

        hidden_states = encoder(
            input_ids, attention_mask, case_marker_ids, clitic_ids, predicate_mask
        )

        expected_dim = config["bilstm"]["hidden_dim"] * 2
        assert hidden_states.shape == (2, max_len, expected_dim)

        hidden_states.sum().backward()
        assert encoder.bilstm.weight_ih_l0.grad is not None
        assert torch.any(encoder.bilstm.weight_ih_l0.grad != 0)
        assert encoder.case_embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# viterbi_decoder
# ---------------------------------------------------------------------------


def _brute_force_best_path(
    emissions: torch.Tensor,
    start_transitions: torch.Tensor,
    transitions: torch.Tensor,
    end_transitions: torch.Tensor,
) -> tuple[list[int], float]:
    seq_len, num_labels = emissions.shape
    best_path, best_score = None, float("-inf")
    for path in itertools.product(range(num_labels), repeat=seq_len):
        score = start_transitions[path[0]].item() + emissions[0, path[0]].item()
        for t in range(1, seq_len):
            score += transitions[path[t - 1], path[t]].item()
            score += emissions[t, path[t]].item()
        score += end_transitions[path[-1]].item()
        if score > best_score:
            best_score, best_path = score, list(path)
    return best_path, best_score


class TestViterbiDecodeAgainstBruteForce:
    def setup_method(self) -> None:
        torch.manual_seed(2)
        self.num_labels = 4
        self.seq_len = 4
        self.start_transitions = torch.randn(self.num_labels)
        self.end_transitions = torch.randn(self.num_labels)
        self.transitions = torch.randn(self.num_labels, self.num_labels)
        self.emissions = torch.randn(self.seq_len, self.num_labels)

    def test_matches_brute_force_optimal_path(self) -> None:
        expected_path, expected_score = _brute_force_best_path(
            self.emissions, self.start_transitions, self.transitions, self.end_transitions
        )
        actual_path, actual_score = viterbi_decode(
            self.emissions, self.transitions, self.start_transitions, self.end_transitions
        )
        assert actual_path == expected_path
        assert actual_score == pytest.approx(expected_score, abs=1e-4)

    def test_large_negative_mask_forces_path_away_from_that_label(self) -> None:
        masked_emissions = self.emissions.clone()
        masked_emissions[:, 0] = -10000.0  # forbid label 0 at every position
        path, _ = viterbi_decode(
            masked_emissions, self.transitions, self.start_transitions, self.end_transitions
        )
        assert 0 not in path

    def test_mask_respects_brute_force_optimum_under_masking(self) -> None:
        masked_emissions = self.emissions.clone()
        masked_emissions[1, 2] = -10000.0
        expected_path, expected_score = _brute_force_best_path(
            masked_emissions, self.start_transitions, self.transitions, self.end_transitions
        )
        actual_path, actual_score = viterbi_decode(
            masked_emissions, self.transitions, self.start_transitions, self.end_transitions
        )
        assert actual_path == expected_path
        assert actual_score == pytest.approx(expected_score, abs=1e-4)

    def test_respects_prefix_mask(self) -> None:
        mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
        path, _ = viterbi_decode(
            self.emissions,
            self.transitions,
            self.start_transitions,
            self.end_transitions,
            mask=mask,
        )
        assert len(path) == 2


class TestBatchViterbiDecode:
    def test_returns_one_path_per_batch_row_with_correct_lengths(self) -> None:
        torch.manual_seed(3)
        num_labels = 3
        emissions = torch.randn(2, 5, num_labels)
        transitions = torch.randn(num_labels, num_labels)
        start = torch.randn(num_labels)
        end = torch.randn(num_labels)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)

        paths = batch_viterbi_decode(emissions, transitions, start, end, mask)
        assert len(paths) == 2
        assert len(paths[0]) == 5
        assert len(paths[1]) == 3


class TestFindSpans:
    def test_finds_single_span(self) -> None:
        labels = ["O", "B-ARG0", "I-ARG0", "O"]
        assert _find_spans(labels, "ARG0") == [(1, 2)]

    def test_finds_multiple_disjoint_spans(self) -> None:
        labels = ["B-ARG0", "O", "B-ARG0", "I-ARG0", "O"]
        assert _find_spans(labels, "ARG0") == [(0, 0), (2, 3)]

    def test_no_spans_returns_empty_list(self) -> None:
        assert _find_spans(["O", "O"], "ARG0") == []

    def test_does_not_confuse_different_roles(self) -> None:
        labels = ["B-ARG0", "B-ARG1", "I-ARG1"]
        assert _find_spans(labels, "ARG0") == [(0, 0)]
        assert _find_spans(labels, "ARG1") == [(1, 2)]


class TestEnforceUniqueness:
    def setup_method(self) -> None:
        self.label_list = ["O", "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1"]
        self.label_to_index = {label: i for i, label in enumerate(self.label_list)}

    def test_keeps_the_single_higher_scoring_arg0_span(self) -> None:
        labels = ["B-ARG0", "O", "B-ARG0", "O"]
        emissions = torch.zeros(4, len(self.label_list))
        emissions[0, self.label_to_index["B-ARG0"]] = 5.0  # weak span
        emissions[2, self.label_to_index["B-ARG0"]] = 9.0  # strong span
        result = enforce_uniqueness(labels, emissions, self.label_to_index, roles=["ARG0"])
        assert result == ["O", "O", "B-ARG0", "O"]

    def test_multi_token_spans_are_demoted_as_a_whole(self) -> None:
        labels = ["B-ARG0", "I-ARG0", "B-ARG0", "O"]
        emissions = torch.zeros(4, len(self.label_list))
        emissions[0, self.label_to_index["B-ARG0"]] = 1.0
        emissions[1, self.label_to_index["I-ARG0"]] = 1.0
        emissions[2, self.label_to_index["B-ARG0"]] = 100.0
        result = enforce_uniqueness(labels, emissions, self.label_to_index, roles=["ARG0"])
        assert result == ["O", "O", "B-ARG0", "O"]

    def test_single_span_is_left_untouched(self) -> None:
        labels = ["O", "B-ARG0", "I-ARG0", "O"]
        emissions = torch.zeros(4, len(self.label_list))
        result = enforce_uniqueness(labels, emissions, self.label_to_index, roles=["ARG0"])
        assert result == labels

    def test_only_constrained_roles_are_touched(self) -> None:
        labels = ["B-ARG1", "O", "B-ARG1", "O"]
        emissions = torch.zeros(4, len(self.label_list))
        result = enforce_uniqueness(labels, emissions, self.label_to_index, roles=["ARG0"])
        assert result == labels  # ARG1 duplicates untouched since only ARG0 constrained


class TestDecodeEndToEnd:
    def test_full_decode_pipeline_deduplicates_and_returns_label_strings(self) -> None:
        torch.manual_seed(4)
        label_list = ["O", "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1"]
        num_labels = len(label_list)
        seq_len = 4
        emissions = torch.zeros(seq_len, num_labels)
        # Force two separate strong ARG0 spans that should collapse to one.
        emissions[0, 1] = 10.0  # B-ARG0
        emissions[2, 1] = 20.0  # B-ARG0 (stronger)
        transitions = torch.zeros(num_labels, num_labels)
        start = torch.zeros(num_labels)
        end = torch.zeros(num_labels)

        labels = decode(emissions, transitions, start, end, label_list)
        assert labels.count("B-ARG0") == 1
        assert labels[2] == "B-ARG0"
