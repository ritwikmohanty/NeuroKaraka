"""Runtime generation of the Ontology Masking Matrix M (Section 2.4 of context.txt).

Given a word-level sentence and its predicate lemma, builds an
``(N, |R|)`` additive mask that drives forbidden (token, role) emission
scores toward ``mask_negative_value`` (-10000.0) before the CRF layer adds
it to the neural emissions: ``Phi = Psi_neural + M``.

Rules encoded (from ``config/ontology_rules.yaml``):

1. **Animacy restriction**: if the predicate is an intentional transitive
   verb and a token fails :meth:`HypernymEngine.is_animate`, its B-/I-ARG0
   entries are masked.
2. **Spatial conformance**: if a token has neither a locative postposition
   (में/पर/तक) nor an IndoWordNet [स्थान/Location] sense, its B-/I-ARGM-LOC
   entries are masked.
3. **Temporal conformance**: if a token fails :meth:`HypernymEngine.is_time`,
   its B-/I-ARGM-TMP entries are masked.

Global uniqueness constraints (single ARG0/ARG1 per frame) are a decode-time
structural constraint, not a per-token mask, and are enforced instead by
``src/inference/viterbi_decoder.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import yaml

from src.ontology.hypernym_engine import HypernymEngine

_ONTOLOGY_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ontology_rules.yaml"
)
_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"
)


class MaskGenerator:
    """Builds the ontology masking matrix M for one (sentence, predicate) pair."""

    def __init__(
        self,
        hypernym_engine: Optional[HypernymEngine] = None,
        ontology_rules_path: Path = _ONTOLOGY_RULES_PATH,
        model_config_path: Path = _MODEL_CONFIG_PATH,
    ) -> None:
        self._engine = hypernym_engine or HypernymEngine()

        with open(ontology_rules_path, encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)
        with open(model_config_path, encoding="utf-8") as fh:
            model_config = yaml.safe_load(fh)

        self._intentional_verbs: Set[str] = set(rules["intentional_transitive_verbs"])
        self._restrictions: Dict[str, bool] = rules["selectional_restrictions"]
        self._locative_markers: Set[str] = {
            postposition
            for postposition, entry in rules["case_markers"].items()
            if postposition != "null_case"
            and "ARGM-LOC" in entry.get("roles_favored", [])
        }
        # Postpositions are grammatical particles, not nominal content words:
        # HDTB attaches a postposition to its noun as part of the same chunk
        # (e.g. "बच्चे ने" -> one ARG0 span, both tokens tagged I-ARG0), so a
        # postposition token can itself land in an argument span. It has no
        # real animacy/location/time property, so the selectional-restriction
        # gates below must skip it rather than spuriously masking it (which
        # would otherwise mask the gold label of every multi-token chunk).
        self._postposition_lemmas: Set[str] = {
            postposition
            for postposition in rules["case_markers"]
            if postposition != "null_case"
        }

        self._labels: List[str] = model_config["crf"]["label_set"]
        self._label_to_index: Dict[str, int] = {
            label: idx for idx, label in enumerate(self._labels)
        }
        self._mask_value: float = model_config["crf"]["mask_negative_value"]

    def _mask_role(self, matrix: torch.Tensor, token_index: int, role: str) -> None:
        for prefix in ("B-", "I-"):
            label = f"{prefix}{role}"
            label_index = self._label_to_index.get(label)
            if label_index is not None:
                matrix[token_index, label_index] = self._mask_value

    def generate(
        self,
        lemmas: List[str],
        predicate_lemma: str,
        case_markers: Optional[List[Optional[str]]] = None,
    ) -> torch.Tensor:
        """Build the ``(len(lemmas), |R|)`` additive mask M for this frame.

        Args:
            lemmas: word-level lemma for each token in the sentence.
            predicate_lemma: lemma of the active predicate for this frame.
            case_markers: word-level case postposition (or None) per token,
                e.g. from ``MorphologyExtractor``; used for the locative gate.
        """
        n = len(lemmas)
        case_markers = case_markers or [None] * n
        matrix = torch.zeros(n, len(self._labels))

        predicate_requires_animate_agent = predicate_lemma in self._intentional_verbs

        for i, lemma in enumerate(lemmas):
            if lemma in self._postposition_lemmas:
                continue

            if (
                self._restrictions.get("arg0_requires_animate", False)
                and predicate_requires_animate_agent
                and not self._engine.is_animate(lemma)
            ):
                self._mask_role(matrix, i, "ARG0")

            if self._restrictions.get(
                "argm_loc_requires_location_or_locative_case", False
            ):
                has_locative_case = case_markers[i] in self._locative_markers
                if not (has_locative_case or self._engine.is_location(lemma)):
                    self._mask_role(matrix, i, "ARGM-LOC")

            if self._restrictions.get(
                "argm_tmp_requires_temporal_entity", False
            ) and not self._engine.is_time(lemma):
                self._mask_role(matrix, i, "ARGM-TMP")

        return matrix
