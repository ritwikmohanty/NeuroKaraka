"""Hindi case-postposition and clitic detection for morphological features.

Hindi marks grammatical case analytically: case postpositions (ने, को, से,
का/के/की, में, पर, तक) are separate word tokens that attach to the nominal
they govern rather than word-internal suffixes. This module detects those
postpositions (and a small set of discourse/focus clitics) in an
already-tokenized sentence and maps each token to the categorical IDs
declared in ``config/ontology_rules.yaml``, for use as morphological
embedding lookup indices (Section 2.1, ``m_i``, of context.txt).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ontology_rules.yaml"
)


@dataclass(frozen=True)
class TokenMorphology:
    """Morphological feature assignment for a single token in a sentence."""

    token: str
    case_marker_id: int
    case_marker: Optional[str]
    is_postposition: bool
    clitic_id: int
    clitic: Optional[str]


class MorphologyExtractor:
    """Detects case postpositions and clitics and maps them to categorical IDs."""

    def __init__(self, rules_path: Path = _CONFIG_PATH) -> None:
        case_markers, clitics = self._load_rules(rules_path)
        self._case_markers: Dict[str, int] = case_markers
        self._clitics: Dict[str, int] = clitics

    @staticmethod
    def _load_rules(rules_path: Path) -> tuple[Dict[str, int], Dict[str, int]]:
        with open(rules_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        case_markers = {
            postposition: entry["id"]
            for postposition, entry in data["case_markers"].items()
            if postposition != "null_case"
        }
        clitics = {
            clitic: entry["id"]
            for clitic, entry in data["clitics"].items()
            if clitic != "null_clitic"
        }
        return case_markers, clitics

    def extract(self, tokens: List[str]) -> List[TokenMorphology]:
        """Return per-token morphology for a word-tokenized sentence.

        A case postposition's ID is assigned both to the postposition token
        itself and, when unmarked, propagated back to the immediately
        preceding token (the nominal head it governs) so the encoder can read
        case off the noun's own hidden state as well as the marker's.
        """
        case_ids: List[int] = [0] * len(tokens)
        case_labels: List[Optional[str]] = [None] * len(tokens)
        is_postposition: List[bool] = [False] * len(tokens)
        clitic_ids: List[int] = [0] * len(tokens)
        clitic_labels: List[Optional[str]] = [None] * len(tokens)

        for i, token in enumerate(tokens):
            case_id = self._case_markers.get(token)
            if case_id is not None:
                case_ids[i] = case_id
                case_labels[i] = token
                is_postposition[i] = True
                if i > 0 and case_ids[i - 1] == 0:
                    case_ids[i - 1] = case_id
                    case_labels[i - 1] = token

            clitic_id = self._clitics.get(token)
            if clitic_id is not None:
                clitic_ids[i] = clitic_id
                clitic_labels[i] = token

        return [
            TokenMorphology(
                token=tokens[i],
                case_marker_id=case_ids[i],
                case_marker=case_labels[i],
                is_postposition=is_postposition[i],
                clitic_id=clitic_ids[i],
                clitic=clitic_labels[i],
            )
            for i in range(len(tokens))
        ]

    def case_ids(self, tokens: List[str]) -> List[int]:
        """Convenience accessor: just the per-token case-marker IDs."""
        return [m.case_marker_id for m in self.extract(tokens)]

    def clitic_ids(self, tokens: List[str]) -> List[int]:
        """Convenience accessor: just the per-token clitic IDs."""
        return [m.clitic_id for m in self.extract(tokens)]
