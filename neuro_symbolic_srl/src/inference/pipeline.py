"""End-to-end string-to-frame inference engine.

Wires together every Phase 1-4 component into one callable:
tokenize -> morphology -> encoder -> role scorer / WSD head -> ontology mask
-> CRF-combined emissions -> constrained Viterbi decode -> word-level frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import pyiwn
import torch
import yaml
from transformers import AutoTokenizer

from src.data_pipeline.morphology_extractor import MorphologyExtractor
from src.inference.viterbi_decoder import decode, subword_labels_to_word_labels
from src.models.crf_layer import LinearChainCRF
from src.models.encoder import SRLEncoder, load_model_config
from src.models.role_scorer import RoleScorer, pool_predicate_hidden_state
from src.models.wsd_head import WSDHead
from src.ontology.gloss_cache import GlossEmbeddingCache
from src.ontology.hypernym_engine import HypernymEngine
from src.ontology.indowordnet_client import IndoWordNetClient
from src.ontology.mask_generator import MaskGenerator

_ONTOLOGY_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ontology_rules.yaml"
)


class Argument(TypedDict):
    span: List[int]
    text: str
    role: str


class Frame(TypedDict):
    predicate: Dict[str, object]
    arguments: List[Argument]
    word_labels: List[str]


def _extract_arguments(tokens: List[str], word_labels: List[str]) -> List[Argument]:
    arguments: List[Argument] = []
    i, n = 0, len(word_labels)
    while i < n:
        tag = word_labels[i]
        if tag.startswith("B-"):
            role = tag[2:]
            start = i
            j = i + 1
            while j < n and word_labels[j] == f"I-{role}":
                j += 1
            arguments.append(
                {
                    "span": [start, j - 1],
                    "text": " ".join(tokens[start:j]),
                    "role": role,
                }
            )
            i = j
        else:
            i += 1
    return arguments


class SRLPipeline:
    """Loads every model/ontology component once and exposes :meth:`parse`."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_config = load_model_config()

        with open(_ONTOLOGY_RULES_PATH, encoding="utf-8") as fh:
            self.karaka_to_propbank = yaml.safe_load(fh)["karaka_to_propbank"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["encoder"]["backbone_name"]
        )
        self.morphology_extractor = MorphologyExtractor()

        self.iwn_client = IndoWordNetClient()
        self.hypernym_engine = HypernymEngine(client=self.iwn_client)
        self.mask_generator = MaskGenerator(hypernym_engine=self.hypernym_engine)
        self.gloss_cache = GlossEmbeddingCache(iwn_client=self.iwn_client)

        self.label_list: List[str] = self.model_config["crf"]["label_set"]

        self.encoder = SRLEncoder(self.model_config).to(self.device)
        self.role_scorer = RoleScorer(
            hidden_dim=self.encoder.output_dim,
            mlp_hidden_dim=self.model_config["role_scorer"]["mlp_hidden_dim"],
            num_labels=len(self.label_list),
            dropout=self.model_config["role_scorer"]["dropout"],
        ).to(self.device)
        self.wsd_head = WSDHead(
            hidden_dim=self.encoder.output_dim,
            gloss_dim=self.gloss_cache.embedding_dim,
            projection_dim=self.model_config["wsd_head"]["projection_dim"],
        ).to(self.device)
        self.crf = LinearChainCRF(num_labels=len(self.label_list)).to(self.device)

    def load_checkpoint(self, path: Path) -> None:
        """Load weights saved by ``scripts/train.py``'s ``SRLModel.state_dict()``.

        ``SRLModel`` and this pipeline both name their submodules ``encoder``,
        ``role_scorer``, ``wsd_head``, ``crf``, so the checkpoint's prefixed
        keys are simply regrouped and loaded into the matching submodule.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        submodules = {
            "encoder": self.encoder,
            "role_scorer": self.role_scorer,
            "wsd_head": self.wsd_head,
            "crf": self.crf,
        }
        for name, module in submodules.items():
            prefix = f"{name}."
            sub_state = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(sub_state)

    def parse(
        self,
        tokens: List[str],
        lemmas: List[str],
        predicate_index: int,
    ) -> Frame:
        """Parse one word-tokenized, lemmatized sentence around its predicate."""
        predicate_lemma = lemmas[predicate_index]

        morph = self.morphology_extractor.extract(tokens)
        case_markers = [m.case_marker for m in morph]
        word_case_ids = [m.case_marker_id for m in morph]
        word_clitic_ids = [m.clitic_id for m in morph]

        encoding = self.tokenizer(tokens, is_split_into_words=True)
        word_ids = encoding.word_ids()

        input_ids = torch.tensor([encoding["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoding["attention_mask"]], device=self.device)
        case_marker_ids = torch.tensor(
            [[0 if wid is None else word_case_ids[wid] for wid in word_ids]],
            device=self.device,
        )
        clitic_ids = torch.tensor(
            [[0 if wid is None else word_clitic_ids[wid] for wid in word_ids]],
            device=self.device,
        )
        predicate_mask = torch.tensor(
            [[1 if wid == predicate_index else 0 for wid in word_ids]],
            device=self.device,
        )

        with torch.no_grad():
            hidden_states = self.encoder(
                input_ids, attention_mask, case_marker_ids, clitic_ids, predicate_mask
            )
            neural_emissions = self.role_scorer(hidden_states, predicate_mask)

            predicate_hidden = pool_predicate_hidden_state(hidden_states, predicate_mask)
            sense_info = self._disambiguate_sense(predicate_lemma, predicate_hidden)

        word_level_mask = self.mask_generator.generate(
            lemmas, predicate_lemma, case_markers
        )
        subword_mask = torch.zeros(
            1, len(word_ids), len(self.label_list), device=self.device
        )
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                subword_mask[0, i] = word_level_mask[word_id]

        phi = self.crf.combine_emissions(neural_emissions, subword_mask)[0]

        subword_labels = decode(
            phi,
            self.crf.transitions,
            self.crf.start_transitions,
            self.crf.end_transitions,
            self.label_list,
            mask=attention_mask[0],
        )
        word_labels = subword_labels_to_word_labels(
            subword_labels, word_ids, len(tokens)
        )
        arguments = _extract_arguments(tokens, word_labels)

        return {
            "predicate": {
                "text": tokens[predicate_index],
                "lemma": predicate_lemma,
                **sense_info,
            },
            "arguments": arguments,
            "word_labels": word_labels,
        }

    def _disambiguate_sense(
        self, predicate_lemma: str, predicate_hidden: torch.Tensor
    ) -> Dict[str, object]:
        max_candidates = self.model_config["wsd_head"]["max_candidate_senses"]
        candidates, gloss_embeddings = self.gloss_cache.embed_candidates(
            predicate_lemma, pyiwn.PosTag.VERB, max_candidates=max_candidates
        )
        if not candidates:
            return {"sense": None, "synset_id": None}

        gloss_embeddings = gloss_embeddings.unsqueeze(0).to(self.device)
        candidate_mask = torch.ones(1, len(candidates), device=self.device)
        alpha, _ = self.wsd_head(predicate_hidden, gloss_embeddings, candidate_mask)
        best_index = int(torch.argmax(alpha, dim=-1).item())
        best_synset = candidates[best_index]
        return {"sense": best_synset.gloss(), "synset_id": best_synset.synset_id()}
