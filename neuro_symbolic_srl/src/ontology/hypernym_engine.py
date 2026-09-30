"""Semantic selectional-restriction checks grounded in IndoWordNet.

Implements the Boolean predicates from Section 4/Phase 4 of the
specification (``is_animate``, ``is_location``, ``is_time``,
``is_instrument``) by combining two signals:

1. Hypernym-chain traversal over pyiwn's ``HYPERNYMY`` synset relation
   (cycle-safe, depth-capped).
2. Keyword matching of ``config/ontology_rules.yaml``'s
   ``hypernym_categories`` lists against the gloss + lemma text of a
   synset and every node in its hypernym chain.

Signal (2) is necessary because the bundled pyiwn 0.0.5 data only records a
HYPERNYMY relation for a small minority of synsets, so chain traversal alone
is insufficient to reach a root category node for most common nouns.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

import pyiwn
import yaml

from src.ontology.indowordnet_client import IndoWordNetClient

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ontology_rules.yaml"
)
_MAX_HYPERNYM_DEPTH = 15


class HypernymEngine:
    """Queries IndoWordNet to answer coarse semantic-class questions about lemmas."""

    def __init__(
        self,
        client: Optional[IndoWordNetClient] = None,
        rules_path: Path = _CONFIG_PATH,
    ) -> None:
        self._client = client or IndoWordNetClient()
        self._categories: Dict[str, List[str]] = self._load_categories(rules_path)
        self._animate_agentive_patterns: List[Pattern[str]] = (
            self._load_animate_agentive_patterns(rules_path)
        )
        self._chain_cache: Dict[int, List["pyiwn.iwn.Synset"]] = {}
        self._category_cache: Dict[Tuple[str, object, str], bool] = {}

    @staticmethod
    def _load_categories(rules_path: Path) -> Dict[str, List[str]]:
        with open(rules_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data["hypernym_categories"]

    @staticmethod
    def _load_animate_agentive_patterns(rules_path: Path) -> List[Pattern[str]]:
        with open(rules_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        patterns = data.get("animate_agentive_patterns", [])
        return [re.compile(pattern) for pattern in patterns]

    def hypernym_chain(self, synset: "pyiwn.iwn.Synset") -> List["pyiwn.iwn.Synset"]:
        """Walk the HYPERNYMY relation from ``synset`` up to its recorded root.

        Cycle-safe (stops if a synset id repeats) and capped at
        ``_MAX_HYPERNYM_DEPTH`` hops. Returns an empty list when pyiwn has no
        HYPERNYMY data for this synset (common in the free data dump).
        """
        synset_id = synset.synset_id()
        cached = self._chain_cache.get(synset_id)
        if cached is not None:
            return cached

        chain: List["pyiwn.iwn.Synset"] = []
        seen = {synset_id}
        current = synset
        for _ in range(_MAX_HYPERNYM_DEPTH):
            parents = self._client.hypernyms(current)
            if not parents:
                break
            parent = parents[0]
            parent_id = parent.synset_id()
            if parent_id in seen:
                break
            chain.append(parent)
            seen.add(parent_id)
            current = parent

        self._chain_cache[synset_id] = chain
        return chain

    def _text_pool(self, synset: "pyiwn.iwn.Synset") -> str:
        parts = [synset.gloss(), *synset.lemma_names()]
        for hypernym in self.hypernym_chain(synset):
            parts.append(hypernym.gloss())
            parts.extend(hypernym.lemma_names())
        return " ".join(parts)

    def _matches_category(
        self, lemma: str, category: str, pos: Optional["pyiwn.PosTag"] = None
    ) -> bool:
        cache_key = (lemma, pos, category)
        cached = self._category_cache.get(cache_key)
        if cached is not None:
            return cached

        keywords = self._categories[category]
        patterns = self._animate_agentive_patterns if category == "animate" else []
        result = False
        for synset in self._client.synsets(lemma, pos):
            pool = self._text_pool(synset)
            if any(keyword in pool for keyword in keywords):
                result = True
                break
            if any(pattern.search(pool) for pattern in patterns):
                result = True
                break

        self._category_cache[cache_key] = result
        return result

    def is_animate(self, lemma: str, pos: Optional["pyiwn.PosTag"] = None) -> bool:
        """True if any sense of ``lemma`` is subsumed by [चेतन / Animate]."""
        return self._matches_category(lemma, "animate", pos)

    def is_location(self, lemma: str, pos: Optional["pyiwn.PosTag"] = None) -> bool:
        """True if any sense of ``lemma`` is subsumed by [स्थान / Location]."""
        return self._matches_category(lemma, "location", pos)

    def is_time(self, lemma: str, pos: Optional["pyiwn.PosTag"] = None) -> bool:
        """True if any sense of ``lemma`` is subsumed by [समय / Time]."""
        return self._matches_category(lemma, "time", pos)

    def is_instrument(self, lemma: str, pos: Optional["pyiwn.PosTag"] = None) -> bool:
        """True if any sense of ``lemma`` is subsumed by [उपकरण / Instrument]."""
        return self._matches_category(lemma, "instrument", pos)
