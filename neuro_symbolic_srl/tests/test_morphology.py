"""Unit tests for morphology extraction, CoNLL-U parsing, and Karaka->BIO alignment.

Also covers dataset tokenization/collation once torch + the tokenizer are
available (TestSRLDataset), verifying tensor shapes and clitic/case-ID
alignment against a real Indic subword tokenizer.
"""
from __future__ import annotations

import pytest
import torch

from src.data_pipeline.dataset import SRLCollator, SRLDataset, SRLExample
from src.data_pipeline.hdtb_parser import (
    ConllSentence,
    _extract_karaka,
    build_examples_from_conllu,
    find_predicate_index,
    parse_conllu,
    sentence_to_bio,
)
from src.data_pipeline.lemmatizer import guess_noun_lemma, guess_verb_lemma
from src.data_pipeline.morphology_extractor import MorphologyExtractor
from src.ontology.indowordnet_client import IndoWordNetClient

# ---------------------------------------------------------------------------
# MorphologyExtractor
# ---------------------------------------------------------------------------


class TestMorphologyExtractorCaseMarkers:
    def setup_method(self) -> None:
        self.extractor = MorphologyExtractor()

    def test_ergative_ne_marks_itself_and_preceding_noun(self) -> None:
        tokens = ["बच्चे", "ने", "खाया"]
        morph = self.extractor.extract(tokens)
        assert morph[1].is_postposition is True
        assert morph[1].case_marker == "ने"
        assert morph[0].case_marker_id == morph[1].case_marker_id
        assert morph[0].case_marker_id != 0

    def test_accusative_ko_and_instrumental_se_get_distinct_ids(self) -> None:
        tokens = ["सेब", "को", "चाकू", "से", "काटा"]
        ids = self.extractor.case_ids(tokens)
        ko_id = ids[1]
        se_id = ids[3]
        assert ko_id != 0 and se_id != 0
        assert ko_id != se_id
        assert ids[0] == ko_id  # सेब inherits को's case
        assert ids[2] == se_id  # चाकू inherits से's case

    def test_genitive_variants_ka_ke_ki_have_distinct_ids(self) -> None:
        extractor = self.extractor
        ka_id = extractor.case_ids(["राम", "का", "घर"])[1]
        ke_id = extractor.case_ids(["राम", "के", "घर"])[1]
        ki_id = extractor.case_ids(["राम", "की", "किताब"])[1]
        assert len({ka_id, ke_id, ki_id}) == 3
        assert 0 not in {ka_id, ke_id, ki_id}

    def test_locative_markers_mein_par_tak(self) -> None:
        assert self.extractor.case_ids(["घर", "में"])[1] != 0
        assert self.extractor.case_ids(["मेज़", "पर"])[1] != 0
        assert self.extractor.case_ids(["स्कूल", "तक"])[1] != 0

    def test_no_postposition_yields_null_case_for_all_tokens(self) -> None:
        ids = self.extractor.case_ids(["राम", "खाया"])
        assert ids == [0, 0]

    def test_case_id_not_overwritten_by_a_later_unrelated_marker(self) -> None:
        # बच्चे already carries ने's case; a following, unrelated को should not
        # steal the preceding token's already-assigned marker.
        tokens = ["बच्चे", "ने", "सेब", "को", "खाया"]
        ids = self.extractor.case_ids(tokens)
        ne_id = self.extractor._case_markers["ने"]
        ko_id = self.extractor._case_markers["को"]
        assert ids[0] == ne_id
        assert ids[2] == ko_id


class TestMorphologyExtractorClitics:
    def setup_method(self) -> None:
        self.extractor = MorphologyExtractor()

    def test_emphatic_hi_is_detected(self) -> None:
        morph = self.extractor.extract(["उसने", "ही", "किया"])
        assert morph[1].clitic == "ही"
        assert morph[1].clitic_id != 0

    def test_additive_bhi_is_detected(self) -> None:
        ids = self.extractor.clitic_ids(["वह", "भी", "आया"])
        assert ids[1] != 0

    def test_no_clitic_yields_zero(self) -> None:
        ids = self.extractor.clitic_ids(["राम", "गया"])
        assert ids == [0, 0]

    def test_case_and_clitic_ids_are_independent_id_spaces(self) -> None:
        # "ने" and "ही" are unrelated markers; a token should not accidentally
        # pick up a clitic id from the case-marker table or vice versa.
        morph = self.extractor.extract(["बच्चे", "ने", "ही", "खाया"])
        assert morph[1].clitic_id == 0
        assert morph[2].case_marker_id == 0
        assert morph[2].clitic_id != 0


# ---------------------------------------------------------------------------
# hdtb_parser: CoNLL-U reading
# ---------------------------------------------------------------------------

_SAMPLE_CONLLU = """\
# sent_id = 1
# text = बच्चे ने सेब खाया
1\tबच्चे\tबच्चा\tNOUN\t_\t_\t4\tk1\t_\t_
2\tने\tने\tADP\t_\t_\t1\tlwg__psp\t_\t_
3\tसेब\tसेब\tNOUN\t_\t_\t4\tk2\t_\t_
4\tखाया\tखाना\tVERB\t_\t_\t0\troot\t_\t_

# sent_id = 2
# text = पुलिस ने दूरबीन से चोर को देखा
1\tपुलिस\tपुलिस\tNOUN\t_\t_\t7\tk1\t_\t_
2\tने\tने\tADP\t_\t_\t1\tlwg__psp\t_\t_
3\tदूरबीन\tदूरबीन\tNOUN\t_\t_\t7\tk3\t_\t_
4\tसे\tसे\tADP\t_\t_\t3\tlwg__psp\t_\t_
5\tचोर\tचोर\tNOUN\t_\t_\t7\tk2\t_\t_
6\tको\tको\tADP\t_\t_\t5\tlwg__psp\t_\t_
7\tदेखा\tदेखना\tVERB\t_\t_\t0\troot\t_\t_
"""


class TestParseConllu:
    def test_parses_two_sentences(self) -> None:
        sentences = parse_conllu(_SAMPLE_CONLLU)
        assert len(sentences) == 2

    def test_first_sentence_tokens_and_heads(self) -> None:
        sentence = parse_conllu(_SAMPLE_CONLLU)[0]
        assert sentence.tokens == ["बच्चे", "ने", "सेब", "खाया"]
        assert sentence.heads == [4, 1, 4, 0]
        assert sentence.deprels[0] == "k1"

    def test_skips_comment_lines(self) -> None:
        sentence = parse_conllu(_SAMPLE_CONLLU)[0]
        assert all(not tok.startswith("#") for tok in sentence.tokens)

    def test_skips_multiword_token_ranges(self) -> None:
        text = "1-2\tकुछ\t_\t_\t_\t_\t_\t_\t_\t_\n1\tक\t_\t_\t_\t_\t0\troot\t_\t_\n2\tुछ\t_\t_\t_\t_\t1\tdep\t_\t_\n"
        sentence = parse_conllu(text)[0]
        assert sentence.tokens == ["क", "ुछ"]

    def test_reads_predicate_synset_id_comment(self) -> None:
        sentence = parse_conllu(_SAMPLE_CONLLU)[0]
        assert sentence.predicate_synset_id is None  # not present in this fixture

        text = (
            "# predicate_synset_id = 10852\n"
            "1\tचाबी\tचाबी\tNOUN\t_\t_\t2\tk1\t_\t_\n"
            "2\tखोला\tखोलना\tVERB\t_\t_\t0\troot\t_\t_\n"
        )
        sentence_with_id = parse_conllu(text)[0]
        assert sentence_with_id.predicate_synset_id == 10852


class TestFindPredicateIndex:
    def test_returns_zero_indexed_root_position(self) -> None:
        sentence = parse_conllu(_SAMPLE_CONLLU)[0]
        assert find_predicate_index(sentence) == 3  # खाया is the 4th token

    def test_returns_none_when_no_root_present(self) -> None:
        sentence = ConllSentence(
            tokens=["a", "b"], lemmas=["a", "b"], upos=["X", "X"],
            heads=[2, 1], deprels=["dep", "dep"],
        )
        assert find_predicate_index(sentence) is None


class TestBuildExamplesFromConllu:
    def test_builds_one_example_per_sentence_with_predicate(self) -> None:
        examples = build_examples_from_conllu(_SAMPLE_CONLLU)
        assert len(examples) == 2

    def test_example_carries_lemmas_and_predicate_lemma(self) -> None:
        examples = build_examples_from_conllu(_SAMPLE_CONLLU)
        first = examples[0]
        assert first.lemmas == ["बच्चा", "ने", "सेब", "खाना"]
        assert first.predicate_lemma == "खाना"
        assert first.predicate_index == 3

    def test_gold_synset_id_is_none_when_comment_absent(self) -> None:
        examples = build_examples_from_conllu(_SAMPLE_CONLLU)
        assert examples[0].predicate_synset_id is None

    def test_gold_synset_id_is_read_from_comment(self) -> None:
        text = (
            "# predicate_synset_id = 13868\n"
            "1\tबच्चे\tबच्चा\tNOUN\t_\t_\t2\tk1\t_\t_\n"
            "2\tखाया\tखाना\tVERB\t_\t_\t0\troot\t_\t_\n"
        )
        examples = build_examples_from_conllu(text)
        assert examples[0].predicate_synset_id == 13868

    def test_bio_tags_match_sentence_to_bio_output(self) -> None:
        examples = build_examples_from_conllu(_SAMPLE_CONLLU)
        expected = sentence_to_bio(parse_conllu(_SAMPLE_CONLLU)[0])
        assert examples[0].bio_tags == expected


class TestExtractKaraka:
    def test_plain_karaka_labels(self) -> None:
        assert _extract_karaka("k1") == "k1"
        assert _extract_karaka("k7p") == "k7p"
        assert _extract_karaka("k7t") == "k7t"

    def test_karaka_with_double_underscore_suffix(self) -> None:
        assert _extract_karaka("k1__") == "k1"

    def test_karaka_with_colon_suffix(self) -> None:
        assert _extract_karaka("k2:name") == "k2"

    def test_non_karaka_relation_returns_none(self) -> None:
        assert _extract_karaka("lwg__psp") is None
        assert _extract_karaka("root") is None
        assert _extract_karaka("nmod") is None


# ---------------------------------------------------------------------------
# hdtb_parser: Karaka -> BIO conversion
# ---------------------------------------------------------------------------


class TestSentenceToBio:
    def test_simple_transitive_sentence(self) -> None:
        # बच्चे+ने form one ARG0 chunk (ने attaches via lwg__psp), सेब is ARG1.
        sentence = parse_conllu(_SAMPLE_CONLLU)[0]
        tags = sentence_to_bio(sentence)
        assert tags == ["B-ARG0", "I-ARG0", "B-ARG1", "O"]

    def test_instrument_attaches_as_argm_ins(self) -> None:
        sentence = parse_conllu(_SAMPLE_CONLLU)[1]
        tags = sentence_to_bio(sentence)
        # पुलिस+ने(ARG0) दूरबीन+से(ARGM-INS) चोर+को(ARG1) देखा(O)
        assert tags == [
            "B-ARG0",
            "I-ARG0",
            "B-ARGM-INS",
            "I-ARGM-INS",
            "B-ARG1",
            "I-ARG1",
            "O",
        ]

    def test_multi_token_span_gets_b_then_i_tags(self) -> None:
        sentence = ConllSentence(
            tokens=["बड़ा", "बच्चा", "ने", "खाया"],
            lemmas=["बड़ा", "बच्चा", "ने", "खाना"],
            upos=["ADJ", "NOUN", "ADP", "VERB"],
            heads=[2, 4, 2, 0],
            deprels=["amod", "k1", "lwg__psp", "root"],
        )
        tags = sentence_to_bio(sentence)
        # बड़ा(amod, child of बच्चा) + बच्चा(k1) + ने(lwg__psp of बच्चा) all fall
        # inside बच्चा's subtree, forming one contiguous ARG0 span; B- marks
        # the leftmost token of the span (बड़ा), not the karaka-bearing head.
        assert tags == ["B-ARG0", "I-ARG0", "I-ARG0", "O"]

    def test_unmapped_karaka_code_is_ignored(self) -> None:
        sentence = ConllSentence(
            tokens=["a", "b"],
            lemmas=["a", "b"],
            upos=["NOUN", "VERB"],
            heads=[2, 0],
            deprels=["k99", "root"],
        )
        tags = sentence_to_bio(sentence)
        assert tags == ["O", "O"]

    def test_empty_sentence_returns_empty_tags(self) -> None:
        assert sentence_to_bio(ConllSentence()) == []


# ---------------------------------------------------------------------------
# SRLDataset / collate_fn (requires torch + a real Indic tokenizer)
# ---------------------------------------------------------------------------

_LABEL_LIST = [
    "O",
    "B-ARG0",
    "I-ARG0",
    "B-ARG1",
    "I-ARG1",
    "B-ARGM-INS",
    "I-ARGM-INS",
]


def _make_example(tokens, tags, predicate_index) -> SRLExample:
    extractor = MorphologyExtractor()
    morph = extractor.extract(tokens)
    return SRLExample(
        tokens=tokens,
        bio_tags=tags,
        predicate_index=predicate_index,
        case_marker_ids=[m.case_marker_id for m in morph],
        clitic_ids=[m.clitic_id for m in morph],
    )


class TestSRLDataset:
    def setup_method(self) -> None:
        self.label_to_id = {label: i for i, label in enumerate(_LABEL_LIST)}
        self.examples = [
            _make_example(
                ["बच्चे", "ने", "सेब", "खाया"],
                ["B-ARG0", "O", "B-ARG1", "O"],
                predicate_index=3,
            ),
            _make_example(
                ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखा"],
                ["B-ARG0", "O", "B-ARGM-INS", "O", "B-ARG1", "O", "O"],
                predicate_index=6,
            ),
        ]
        self.dataset = SRLDataset(self.examples, label_to_id=self.label_to_id)
        self.collator = SRLCollator(
            tokenizer=self.dataset.tokenizer, label_pad_id=-100
        )

    def test_dataset_length(self) -> None:
        assert len(self.dataset) == 2

    def test_getitem_returns_expected_keys(self) -> None:
        item = self.dataset[0]
        for key in (
            "input_ids",
            "attention_mask",
            "word_ids",
            "case_marker_ids",
            "clitic_ids",
            "predicate_mask",
            "label_ids",
        ):
            assert key in item

    def test_predicate_mask_has_exactly_one_hot_subword_group(self) -> None:
        item = self.dataset[0]
        word_ids = item["word_ids"]
        predicate_mask = item["predicate_mask"]
        flagged_word_ids = {
            word_ids[i] for i, flag in enumerate(predicate_mask) if flag == 1
        }
        assert flagged_word_ids == {3}  # खाया is word index 3

    def test_word_level_ids_align_length_with_subword_ids(self) -> None:
        item = self.dataset[0]
        assert len(item["case_marker_ids"]) == len(item["input_ids"])
        assert len(item["clitic_ids"]) == len(item["input_ids"])
        assert len(item["label_ids"]) == len(item["input_ids"])

    def test_special_tokens_get_null_case_and_ignored_label(self) -> None:
        item = self.dataset[0]
        word_ids = item["word_ids"]
        for i, wid in enumerate(word_ids):
            if wid is None:
                assert item["case_marker_ids"][i] == 0
                assert item["label_ids"][i] == -100

    def test_collate_batches_with_dynamic_padding(self) -> None:
        batch = [self.dataset[0], self.dataset[1]]
        collated = self.collator(batch)

        max_len = max(len(ex["input_ids"]) for ex in batch)
        for key in (
            "input_ids",
            "attention_mask",
            "case_marker_ids",
            "clitic_ids",
            "predicate_mask",
            "label_ids",
        ):
            tensor = collated[key]
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (2, max_len)

    def test_collate_pads_shorter_sequence_with_attention_mask_zero(self) -> None:
        batch = [self.dataset[0], self.dataset[1]]
        collated = self.collator(batch)
        shorter_len = len(batch[0]["input_ids"])
        assert collated["attention_mask"][0, shorter_len:].sum().item() == 0

    def test_collate_label_padding_uses_ignore_index(self) -> None:
        batch = [self.dataset[0], self.dataset[1]]
        collated = self.collator(batch)
        shorter_len = len(batch[0]["input_ids"])
        padded_labels = collated["label_ids"][0, shorter_len:]
        assert torch.all(padded_labels == -100)

    def test_each_example_has_exactly_one_predicate_token_per_batch_row(self) -> None:
        batch = [self.dataset[0], self.dataset[1]]
        collated = self.collator(batch)
        # predicate indicator marks a whole subword group, so just assert >=1
        assert (collated["predicate_mask"].sum(dim=1) >= 1).all()


# ---------------------------------------------------------------------------
# lemmatizer: dictionary-validated verb/noun surface-form guessers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iwn_client() -> IndoWordNetClient:
    return IndoWordNetClient()


class TestGuessVerbLemma:
    def test_regular_perfective_forms_resolve_to_infinitive(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        cases = {
            "खोला": "खोलना",
            "देखा": "देखना",
            "पढ़ी": "पढ़ना",
            "सुनी": "सुनना",
            "बोला": "बोलना",
            "मारा": "मारना",
            "लिखा": "लिखना",
            "सोचा": "सोचना",
        }
        for surface, expected in cases.items():
            assert guess_verb_lemma(surface, client=iwn_client) == expected

    def test_glide_insertion_forms_resolve_to_infinitive(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        # खाना/बनाना insert a "य" glide before the perfective matra
        # (खाया, not "खाआ"); these need the "या" suffix, not bare "ा".
        assert guess_verb_lemma("खाया", client=iwn_client) == "खाना"
        assert guess_verb_lemma("बनाया", client=iwn_client) == "बनाना"

    def test_already_infinitive_form_is_returned_unchanged(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        assert guess_verb_lemma("खोलना", client=iwn_client) == "खोलना"

    def test_unrecognized_word_falls_back_to_surface_form(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        assert guess_verb_lemma("zzzznotaword123", client=iwn_client) == "zzzznotaword123"

    def test_suppletive_irregular_perfective_is_a_known_gap(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        # गया (जाना's suppletive perfective) cannot be recovered by suffix
        # stripping — documented limitation, not a silent success.
        assert guess_verb_lemma("गया", client=iwn_client) == "गया"


class TestGuessNounLemma:
    def test_masculine_oblique_forms_resolve_to_direct_case(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        cases = {
            "बच्चे": "बच्चा",
            "लड़के": "लड़का",
            "घोड़े": "घोड़ा",
            "कुत्ते": "कुत्ता",
        }
        for surface, expected in cases.items():
            assert guess_noun_lemma(surface, client=iwn_client) == expected

    def test_already_direct_case_form_is_returned_unchanged(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        assert guess_noun_lemma("चाबी", client=iwn_client) == "चाबी"

    def test_unrecognized_word_falls_back_to_surface_form(
        self, iwn_client: IndoWordNetClient
    ) -> None:
        assert guess_noun_lemma("zzzznotaword123", client=iwn_client) == "zzzznotaword123"
