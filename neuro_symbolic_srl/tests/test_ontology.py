"""Unit tests for the IndoWordNet client and hypernym/selectional-restriction engine.

Exercises the real pyiwn Hindi database (no mocking) against the example
nouns from context.txt: चाबी [inanimate], बच्चा [animate], स्कूल [location],
plus a couple of additional common nouns for broader coverage.
"""
from __future__ import annotations

import pytest
import torch
import yaml

from src.ontology.hypernym_engine import HypernymEngine
from src.ontology.indowordnet_client import IndoWordNetClient
from src.ontology.mask_generator import MaskGenerator, _MODEL_CONFIG_PATH


@pytest.fixture(scope="module")
def client() -> IndoWordNetClient:
    return IndoWordNetClient()


@pytest.fixture(scope="module")
def engine(client: IndoWordNetClient) -> HypernymEngine:
    return HypernymEngine(client=client)


class TestIndoWordNetClient:
    def test_synsets_returns_nonempty_for_known_word(self, client: IndoWordNetClient) -> None:
        synsets = client.synsets("चाबी")
        assert len(synsets) > 0

    def test_synsets_empty_for_unknown_word(self, client: IndoWordNetClient) -> None:
        synsets = client.synsets("zzzznotaword123")
        assert list(synsets) == []

    def test_gloss_and_lemma_names_are_nonempty_strings(
        self, client: IndoWordNetClient
    ) -> None:
        synset = client.synsets("चाबी")[0]
        assert isinstance(client.gloss(synset), str) and client.gloss(synset)
        lemma_names = client.lemma_names(synset)
        assert len(lemma_names) > 0
        assert all(isinstance(name, str) for name in lemma_names)

    def test_hypernyms_returns_a_sequence(self, client: IndoWordNetClient) -> None:
        synset = client.synsets("चाबी")[0]
        hypernyms = client.hypernyms(synset)
        assert isinstance(hypernyms, list)


class TestHypernymEngineAnimacy:
    def test_chaabi_key_is_not_animate(self, engine: HypernymEngine) -> None:
        assert engine.is_animate("चाबी") is False

    def test_bachcha_child_is_animate(self, engine: HypernymEngine) -> None:
        assert engine.is_animate("बच्चा") is True

    def test_school_is_not_animate(self, engine: HypernymEngine) -> None:
        assert engine.is_animate("स्कूल") is False

    def test_thief_is_animate_via_agentive_relative_clause_gloss(
        self, engine: HypernymEngine
    ) -> None:
        # चोर's gloss is "वह जो चोरी करता हो" (one who steals) — no explicit
        # चेतन/मानव/व्यक्ति keyword, only the "वह जो...करता" agentive pattern.
        assert engine.is_animate("चोर") is True

    def test_police_is_animate_via_agentive_vaala_gloss(
        self, engine: HypernymEngine
    ) -> None:
        # पुलिस (सिपाही sense) gloss uses "...रक्षा करने वाला सिपाही..."
        # (one who protects...) — the "-करने वाला" agentive nominalizer.
        assert engine.is_animate("पुलिस") is True


class TestHypernymEngineLocation:
    def test_school_is_location(self, engine: HypernymEngine) -> None:
        assert engine.is_location("स्कूल") is True

    def test_chaabi_key_is_not_location(self, engine: HypernymEngine) -> None:
        assert engine.is_location("चाबी") is False

    def test_bachcha_child_is_not_location(self, engine: HypernymEngine) -> None:
        assert engine.is_location("बच्चा") is False


class TestHypernymEngineTime:
    def test_samay_time_is_time(self, engine: HypernymEngine) -> None:
        assert engine.is_time("समय") is True

    def test_subah_morning_is_time(self, engine: HypernymEngine) -> None:
        assert engine.is_time("सुबह") is True

    def test_chaabi_key_is_not_time(self, engine: HypernymEngine) -> None:
        assert engine.is_time("चाबी") is False


class TestHypernymEngineInstrument:
    def test_chaabi_key_is_instrument(self, engine: HypernymEngine) -> None:
        assert engine.is_instrument("चाबी") is True

    def test_doorbeen_telescope_is_instrument(self, engine: HypernymEngine) -> None:
        assert engine.is_instrument("दूरबीन") is True

    def test_bachcha_child_is_not_instrument(self, engine: HypernymEngine) -> None:
        assert engine.is_instrument("बच्चा") is False


class TestHypernymEngineCaching:
    def test_repeated_lookup_uses_cache_and_is_consistent(
        self, engine: HypernymEngine
    ) -> None:
        first = engine.is_animate("बच्चा")
        second = engine.is_animate("बच्चा")
        assert first == second is True
        cache_key = ("बच्चा", None, "animate")
        assert cache_key in engine._category_cache

    def test_unknown_word_defaults_to_false_for_all_checks(
        self, engine: HypernymEngine
    ) -> None:
        unknown = "zzzznotaword123"
        assert engine.is_animate(unknown) is False
        assert engine.is_location(unknown) is False
        assert engine.is_time(unknown) is False
        assert engine.is_instrument(unknown) is False


class TestHypernymChainTraversal:
    def test_hypernym_chain_is_cycle_safe_and_bounded(
        self, engine: HypernymEngine, client: IndoWordNetClient
    ) -> None:
        synset = client.synsets("चाबी")[0]
        chain = engine.hypernym_chain(synset)
        assert isinstance(chain, list)
        seen_ids = [synset.synset_id()] + [s.synset_id() for s in chain]
        assert len(seen_ids) == len(set(seen_ids))
        assert len(chain) <= 15


# ---------------------------------------------------------------------------
# MaskGenerator (Section 2.4 of context.txt): the runtime Ontology Masking
# Matrix M, built on top of HypernymEngine's selectional-restriction checks.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def label_to_index() -> dict:
    with open(_MODEL_CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    labels = config["crf"]["label_set"]
    return {label: i for i, label in enumerate(labels)}


@pytest.fixture(scope="module")
def mask_generator(engine: HypernymEngine) -> MaskGenerator:
    return MaskGenerator(hypernym_engine=engine)


class TestMaskGeneratorAnimacyGate:
    def test_key_as_ergative_subject_of_intentional_verb_gets_arg0_masked(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        # चाबी ने ताला खोला — चाबी must NOT be a viable ARG0 for खोलना.
        matrix = mask_generator.generate(
            lemmas=["चाबी", "ताला", "खोलना"],
            predicate_lemma="खोलना",
            case_markers=[None, None, None],
        )
        assert matrix[0, label_to_index["B-ARG0"]] == -10000.0
        assert matrix[0, label_to_index["I-ARG0"]] == -10000.0

    def test_animate_subject_of_intentional_verb_keeps_arg0_open(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        # सेब को बच्चे ने खाया — बच्चा must remain a viable ARG0 for खाना.
        matrix = mask_generator.generate(
            lemmas=["सेब", "बच्चा", "खाना"],
            predicate_lemma="खाना",
            case_markers=[None, None, None],
        )
        assert matrix[1, label_to_index["B-ARG0"]] == 0.0
        assert matrix[1, label_to_index["I-ARG0"]] == 0.0

    def test_inanimate_arg0_stays_unmasked_for_non_intentional_predicate(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        # The animacy gate only fires for predicates in
        # intentional_transitive_verbs; गिरना (to fall) is not one of them.
        matrix = mask_generator.generate(
            lemmas=["चाबी"], predicate_lemma="गिरना", case_markers=[None]
        )
        assert matrix[0, label_to_index["B-ARG0"]] == 0.0


class TestMaskGeneratorLocativeGate:
    def test_location_noun_stays_open_without_a_case_marker(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        matrix = mask_generator.generate(
            lemmas=["स्कूल"], predicate_lemma="जाना", case_markers=[None]
        )
        assert matrix[0, label_to_index["B-ARGM-LOC"]] == 0.0

    def test_non_location_noun_without_locative_case_gets_masked(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        matrix = mask_generator.generate(
            lemmas=["चाबी"], predicate_lemma="खोलना", case_markers=[None]
        )
        assert matrix[0, label_to_index["B-ARGM-LOC"]] == -10000.0

    def test_locative_case_marker_alone_keeps_argm_loc_open(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        # चाबी is not a location sense, but a "में" case marker should still
        # satisfy the locative gate per the disjunctive rule in Section 2.4.
        matrix = mask_generator.generate(
            lemmas=["चाबी"], predicate_lemma="खोलना", case_markers=["में"]
        )
        assert matrix[0, label_to_index["B-ARGM-LOC"]] == 0.0


class TestMaskGeneratorTemporalGate:
    def test_temporal_noun_stays_open(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        matrix = mask_generator.generate(
            lemmas=["सुबह"], predicate_lemma="जाना", case_markers=[None]
        )
        assert matrix[0, label_to_index["B-ARGM-TMP"]] == 0.0

    def test_non_temporal_noun_gets_masked(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        matrix = mask_generator.generate(
            lemmas=["चाबी"], predicate_lemma="खोलना", case_markers=[None]
        )
        assert matrix[0, label_to_index["B-ARGM-TMP"]] == -10000.0


class TestMaskGeneratorPostpositionExemption:
    def test_postposition_token_is_not_masked_even_if_it_would_fail_animacy(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        # In HDTB, a postposition shares its noun's argument span (e.g.
        # "बच्चे ने" is one I-ARG0 chunk). ने itself has no animacy, so the
        # ARG0 gate must skip it rather than masking the chunk's own gold
        # continuation label — this previously made training loss explode.
        matrix = mask_generator.generate(
            lemmas=["बच्चा", "ने", "खाना"],
            predicate_lemma="खाना",
            case_markers=[None, "ने", None],
        )
        assert matrix[1, label_to_index["B-ARG0"]] == 0.0
        assert matrix[1, label_to_index["I-ARG0"]] == 0.0


class TestMaskGeneratorShapeAndUnrelatedLabels:
    def test_matrix_shape_matches_sentence_length_and_label_count(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        lemmas = ["पुलिस", "दूरबीन", "चोर", "देखना"]
        matrix = mask_generator.generate(
            lemmas=lemmas, predicate_lemma="देखना", case_markers=[None] * 4
        )
        assert isinstance(matrix, torch.Tensor)
        assert matrix.shape == (len(lemmas), len(label_to_index))

    def test_unrelated_labels_like_arg1_are_never_masked_by_this_generator(
        self, mask_generator: MaskGenerator, label_to_index: dict
    ) -> None:
        matrix = mask_generator.generate(
            lemmas=["चाबी"], predicate_lemma="खोलना", case_markers=[None]
        )
        assert matrix[0, label_to_index["B-ARG1"]] == 0.0
        assert matrix[0, label_to_index["O"]] == 0.0
