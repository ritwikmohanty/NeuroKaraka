"""Presentation demo: shows the ontology mask changing the decoded output live.

For सेंटेंस 1 specifically, this prints BOTH the raw neural-only decode
(Phi = Psi_neural, no mask) and the full neuro-symbolic decode
(Phi = Psi_neural + M), side by side — proving empirically that the
ontology mask is doing necessary work, not just window dressing: the
trained neural network *by itself* still falls into the "surface
ergative-postposition trap" (चाबी -> an ARG0-family label) described in
context.txt Section 1.1, and only the mask corrects it.

Usage:
    python scripts/demo.py
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from src.data_pipeline.morphology_extractor import MorphologyExtractor
from src.inference.viterbi_decoder import decode, subword_labels_to_word_labels
from src.models.crf_layer import LinearChainCRF
from src.models.encoder import SRLEncoder, load_model_config
from src.models.role_scorer import RoleScorer
from src.ontology.hypernym_engine import HypernymEngine
from src.ontology.indowordnet_client import IndoWordNetClient
from src.ontology.mask_generator import MaskGenerator

CHECKPOINT_PATH = "checkpoints/final_model.pt"


def _load_models(model_config):
    label_list = model_config["crf"]["label_set"]
    encoder = SRLEncoder(model_config)
    role_scorer = RoleScorer(
        hidden_dim=encoder.output_dim,
        mlp_hidden_dim=model_config["role_scorer"]["mlp_hidden_dim"],
        num_labels=len(label_list),
        dropout=model_config["role_scorer"]["dropout"],
    )
    crf = LinearChainCRF(num_labels=len(label_list))

    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    encoder.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    role_scorer.load_state_dict(
        {k[len("role_scorer."):]: v for k, v in state.items() if k.startswith("role_scorer.")}
    )
    crf.load_state_dict({k[len("crf."):]: v for k, v in state.items() if k.startswith("crf.")})
    encoder.eval()
    role_scorer.eval()
    return encoder, role_scorer, crf, label_list


def _run_neural(encoder, role_scorer, tokenizer, morphology_extractor, tokens, predicate_index):
    encoding = tokenizer(tokens, is_split_into_words=True)
    word_ids = encoding.word_ids()
    input_ids = torch.tensor([encoding["input_ids"]])
    attention_mask = torch.tensor([encoding["attention_mask"]])

    morph = morphology_extractor.extract(tokens)
    word_case_ids = [m.case_marker_id for m in morph]
    word_clitic_ids = [m.clitic_id for m in morph]
    case_ids = torch.tensor(
        [[0 if wid is None else word_case_ids[wid] for wid in word_ids]], dtype=torch.long
    )
    clitic_ids = torch.tensor(
        [[0 if wid is None else word_clitic_ids[wid] for wid in word_ids]], dtype=torch.long
    )
    predicate_mask = torch.tensor(
        [[1 if wid == predicate_index else 0 for wid in word_ids]], dtype=torch.long
    )

    with torch.no_grad():
        hidden = encoder(input_ids, attention_mask, case_ids, clitic_ids, predicate_mask)
        emissions = role_scorer(hidden, predicate_mask)

    return emissions, attention_mask, word_ids


def _decode_word_labels(emissions, attention_mask, word_ids, crf, label_list, num_words, ontology_mask=None):
    phi = crf.combine_emissions(emissions, ontology_mask)[0] if ontology_mask is not None else emissions[0]
    subword_labels = decode(
        phi, crf.transitions, crf.start_transitions, crf.end_transitions, label_list, mask=attention_mask[0]
    )
    return subword_labels_to_word_labels(subword_labels, word_ids, num_words)


def _print_frame(tokens, word_labels):
    for token, label in zip(tokens, word_labels):
        marker = "  <-- " if label not in ("O",) else ""
        print(f"    {token:10s} {label}{marker}")


def main() -> None:
    print("=" * 72)
    print("Neuro-Symbolic SRL — Live Demo")
    print("=" * 72)

    model_config = load_model_config()
    encoder, role_scorer, crf, label_list = _load_models(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config["encoder"]["backbone_name"])
    morphology_extractor = MorphologyExtractor()
    iwn_client = IndoWordNetClient()
    mask_generator = MaskGenerator(hypernym_engine=HypernymEngine(client=iwn_client))

    # --- Sentence 1: the ablation that proves the mask matters ----------
    print("\n[1] चाबी ने ताला खोला  (\"The key opened the lock\")")
    print("    Animacy-inversion trap: चाबी carries the ergative ने, the\n"
          "    surface cue for an agent — does the model fall for it?\n")

    tokens = ["चाबी", "ने", "ताला", "खोला"]
    lemmas = ["चाबी", "ने", "ताला", "खोलना"]
    emissions, attention_mask, word_ids = _run_neural(
        encoder, role_scorer, tokenizer, morphology_extractor, tokens, 3
    )

    no_mask_labels = _decode_word_labels(
        emissions, attention_mask, word_ids, crf, label_list, len(tokens)
    )
    print("  >> Neural network ALONE (no ontology mask):")
    _print_frame(tokens, no_mask_labels)

    case_markers = [m.case_marker for m in morphology_extractor.extract(tokens)]
    word_mask = mask_generator.generate(lemmas, "खोलना", case_markers)
    subword_mask = torch.zeros(1, emissions.shape[1], len(label_list))
    for i, wid in enumerate(word_ids):
        if wid is not None:
            subword_mask[0, i] = word_mask[wid]

    with_mask_labels = _decode_word_labels(
        emissions, attention_mask, word_ids, crf, label_list, len(tokens), ontology_mask=subword_mask
    )
    print("\n  >> FULL system (+ IndoWordNet ontology mask):")
    _print_frame(tokens, with_mask_labels)

    if any(lab in ("B-ARG0", "I-ARG0") for lab in no_mask_labels) and not any(
        lab in ("B-ARG0", "I-ARG0") for lab in with_mask_labels
    ):
        print("\n  *** The mask changed the outcome: चाबी flipped OUT of the")
        print("      ARG0 family once IndoWordNet confirmed it's inanimate. ***")

    # --- Sentences 2 & 3: full system only ------------------------------
    for title, tokens, lemmas, predicate_lemma, predicate_index in [
        (
            "[2] सेब को बच्चे ने खाया  (\"The apple, the child ate\" — scrambled OSV)",
            ["सेब", "को", "बच्चे", "ने", "खाया"],
            ["सेब", "को", "बच्चा", "ने", "खाना"],
            "खाना",
            4,
        ),
        (
            "[3] पुलिस ने दूरबीन से चोर को देखा  (\"The police saw the thief with a telescope\")",
            ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखा"],
            ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखना"],
            "देखना",
            6,
        ),
    ]:
        print(f"\n{title}\n")
        emissions, attention_mask, word_ids = _run_neural(
            encoder, role_scorer, tokenizer, morphology_extractor, tokens, predicate_index
        )
        case_markers = [m.case_marker for m in morphology_extractor.extract(tokens)]
        word_mask = mask_generator.generate(lemmas, predicate_lemma, case_markers)
        subword_mask = torch.zeros(1, emissions.shape[1], len(label_list))
        for i, wid in enumerate(word_ids):
            if wid is not None:
                subword_mask[0, i] = word_mask[wid]
        labels = _decode_word_labels(
            emissions, attention_mask, word_ids, crf, label_list, len(tokens), ontology_mask=subword_mask
        )
        _print_frame(tokens, labels)

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
