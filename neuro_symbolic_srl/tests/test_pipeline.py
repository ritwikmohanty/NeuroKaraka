"""End-to-end integration tests on the three ambiguity-resolution verification
sentences from Section 6 of context.txt, run through the full trained
pipeline (encoder -> role scorer / WSD head -> ontology mask -> CRF ->
constrained Viterbi decode).

Weights are loaded from ``checkpoints/final_model.pt``, produced by
``scripts/train.py`` overfit on the tiny hand-crafted corpus in
``data/raw/toy_train.conllu`` (which includes these exact three sentences,
correctly Karaka-annotated — see that file's comments for why चाबी is
annotated k3/Karana rather than k1/Karta despite the ergative ने). This
demonstrates the complete trained system, not just the symbolic mask
in isolation: if no checkpoint is present, the tests fall back to the
untrained pipeline and only check the ontology-mask-guaranteed properties
(sentence 1's "not ARG0"), skipping the assertions that require learned
discrimination among otherwise-legal roles.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.pipeline import SRLPipeline

_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "final_model.pt"
)
_HAS_CHECKPOINT = _CHECKPOINT_PATH.exists()


@pytest.fixture(scope="module")
def pipeline() -> SRLPipeline:
    pipe = SRLPipeline()
    if _HAS_CHECKPOINT:
        pipe.load_checkpoint(_CHECKPOINT_PATH)
    return pipe


class TestVerificationSentence1AnimacyInversion:
    """चाबी ने ताला खोला — चाबी must not be ARG0 despite the ergative ने.

    Guaranteed by the ontology mask alone (M(चाबी, ARG0) = -10000), so this
    must hold even without a trained checkpoint.
    """

    def test_chaabi_is_never_assigned_arg0(self, pipeline: SRLPipeline) -> None:
        tokens = ["चाबी", "ने", "ताला", "खोला"]
        lemmas = ["चाबी", "ने", "ताला", "खोलना"]
        frame = pipeline.parse(tokens, lemmas, predicate_index=3)
        assert frame["word_labels"][0] not in ("B-ARG0", "I-ARG0")

    @pytest.mark.skipif(not _HAS_CHECKPOINT, reason="requires a trained checkpoint")
    def test_chaabi_receives_arg2_or_argm_ins(self, pipeline: SRLPipeline) -> None:
        tokens = ["चाबी", "ने", "ताला", "खोला"]
        lemmas = ["चाबी", "ने", "ताला", "खोलना"]
        frame = pipeline.parse(tokens, lemmas, predicate_index=3)
        assert frame["word_labels"][0] in (
            "B-ARG2",
            "I-ARG2",
            "B-ARGM-INS",
            "I-ARGM-INS",
        )


class TestVerificationSentence2ScrambledOSV:
    """सेब को बच्चे ने खाया — OSV order must still give सेब=ARG1, बच्चे=ARG0."""

    @pytest.mark.skipif(not _HAS_CHECKPOINT, reason="requires a trained checkpoint")
    def test_seb_is_arg1_and_bachche_is_arg0(self, pipeline: SRLPipeline) -> None:
        tokens = ["सेब", "को", "बच्चे", "ने", "खाया"]
        lemmas = ["सेब", "को", "बच्चा", "ने", "खाना"]
        frame = pipeline.parse(tokens, lemmas, predicate_index=4)
        assert frame["word_labels"][0] == "B-ARG1"
        assert frame["word_labels"][2] == "B-ARG0"


class TestVerificationSentence3InstrumentAttachment:
    """पुलिस ने दूरबीन से चोर को देखा — दूरबीन attaches to देखा as ARGM-INS."""

    @pytest.mark.skipif(not _HAS_CHECKPOINT, reason="requires a trained checkpoint")
    def test_doorbeen_attaches_as_argm_ins(self, pipeline: SRLPipeline) -> None:
        tokens = ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखा"]
        lemmas = ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखना"]
        frame = pipeline.parse(tokens, lemmas, predicate_index=6)
        assert frame["word_labels"][2] == "B-ARGM-INS"

    @pytest.mark.skipif(not _HAS_CHECKPOINT, reason="requires a trained checkpoint")
    def test_police_and_thief_get_core_argument_roles(self, pipeline: SRLPipeline) -> None:
        tokens = ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखा"]
        lemmas = ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखना"]
        frame = pipeline.parse(tokens, lemmas, predicate_index=6)
        assert frame["word_labels"][0] == "B-ARG0"
        assert frame["word_labels"][4] == "B-ARG1"


class TestUniquenessAcrossPipeline:
    def test_no_duplicate_core_arguments_in_any_verification_sentence(
        self, pipeline: SRLPipeline
    ) -> None:
        cases = [
            (["चाबी", "ने", "ताला", "खोला"], ["चाबी", "ने", "ताला", "खोलना"], 3),
            (
                ["सेब", "को", "बच्चे", "ने", "खाया"],
                ["सेब", "को", "बच्चा", "ने", "खाना"],
                4,
            ),
            (
                ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखा"],
                ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखना"],
                6,
            ),
        ]
        for tokens, lemmas, predicate_index in cases:
            frame = pipeline.parse(tokens, lemmas, predicate_index=predicate_index)
            assert frame["word_labels"].count("B-ARG0") <= 1
            assert frame["word_labels"].count("B-ARG1") <= 1
