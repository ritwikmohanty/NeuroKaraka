"""CoNLL-U reader for HDTB / UP_Hindi and Karaka -> PropBank BIO alignment.

The Hindi Dependency Treebank encodes Paninian Karaka relations (k1, k2, k3,
k4, k5, k7p, k7t, ...) directly in the DEPREL column, sometimes suffixed with
extra sub-relation markup (e.g. ``k1__``, ``k7p:...``, ``k2-mng``). This
module parses the 10-column CoNLL-U format into ``ConllSentence`` objects and
converts the Karaka dependency labels into span-level BIO PropBank tags per
Section 5 Phase 2 of context.txt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.data_pipeline.dataset import SRLExample
from src.data_pipeline.morphology_extractor import MorphologyExtractor

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ontology_rules.yaml"
)
_KARAKA_LABEL_PATTERN = re.compile(r"^(k[0-9]+[a-z]*)", re.IGNORECASE)


@dataclass
class ConllSentence:
    """A single CoNLL-U sentence, 1-indexed to match the HEAD column convention."""

    tokens: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    upos: List[str] = field(default_factory=list)
    heads: List[int] = field(default_factory=list)  # 0 = root
    deprels: List[str] = field(default_factory=list)
    predicate_synset_id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.tokens)


def load_karaka_to_propbank(rules_path: Path = _CONFIG_PATH) -> Dict[str, str]:
    with open(rules_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {k.lower(): v for k, v in data["karaka_to_propbank"].items()}


_SYNSET_ID_COMMENT_PATTERN = re.compile(r"^#\s*predicate_synset_id\s*=\s*(\d+)\s*$")


def parse_conllu(text: str) -> List[ConllSentence]:
    """Parse raw CoNLL-U text into a list of sentences.

    Comment lines (``# ...``) are skipped, except for the optional
    ``# predicate_synset_id = <int>`` metadata line this project uses to
    carry gold IndoWordNet WSD supervision alongside a sentence. Multiword-
    token ranges (IDs like ``3-4``) and empty nodes (IDs like ``3.1``) are
    skipped, matching the standard CoNLL-U convention of only scoring/using
    single-integer IDs.
    """
    sentences: List[ConllSentence] = []
    current = ConllSentence()

    for raw_line in text.splitlines():
        line = raw_line.strip("\n")
        if not line.strip():
            if len(current):
                sentences.append(current)
                current = ConllSentence()
            continue
        if line.startswith("#"):
            match = _SYNSET_ID_COMMENT_PATTERN.match(line)
            if match:
                current.predicate_synset_id = int(match.group(1))
            continue

        columns = line.split("\t")
        if len(columns) < 8:
            continue
        token_id = columns[0]
        if not re.fullmatch(r"[0-9]+", token_id):
            # multiword span ("3-4") or empty node ("3.1"): skip
            continue

        form, lemma, upos = columns[1], columns[2], columns[3]
        head_raw, deprel = columns[6], columns[7]
        head = int(head_raw) if re.fullmatch(r"[0-9]+", head_raw) else 0

        current.tokens.append(form)
        current.lemmas.append(lemma)
        current.upos.append(upos)
        current.heads.append(head)
        current.deprels.append(deprel)

    if len(current):
        sentences.append(current)

    return sentences


def _extract_karaka(deprel: str) -> Optional[str]:
    """Normalize a raw DEPREL field to a base Karaka code, if it is one."""
    base = deprel.split(":")[0].split("__")[0].split("-")[0].strip().lower()
    match = _KARAKA_LABEL_PATTERN.match(base)
    return match.group(1) if match else None


def _build_children_map(heads: List[int]) -> Dict[int, List[int]]:
    """Map each 1-indexed token index to the list of its direct dependents."""
    children: Dict[int, List[int]] = {}
    for i, head in enumerate(heads, start=1):
        children.setdefault(head, []).append(i)
    return children


def _subtree_indices(root: int, children: Dict[int, List[int]]) -> List[int]:
    """Return all 1-indexed descendants of ``root`` (inclusive), sorted."""
    stack = [root]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(children.get(node, []))
    return sorted(visited)


def sentence_to_bio(
    sentence: ConllSentence,
    karaka_to_propbank: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Convert a parsed sentence's Karaka dependencies into BIO PropBank tags.

    For every token whose DEPREL carries a mapped Karaka code, its full
    dependency subtree (itself plus all transitive dependents) becomes the
    argument span: the lowest-indexed token in that span is tagged
    ``B-<ROLE>`` and the rest ``I-<ROLE>``. Tokens outside any argument span
    (including predicate heads) are tagged ``O`` — the predicate is signalled
    separately via the predicate-indicator feature, not a BIO label.
    """
    karaka_to_propbank = karaka_to_propbank or load_karaka_to_propbank()
    n = len(sentence)
    tags: List[str] = ["O"] * n
    children = _build_children_map(sentence.heads)

    for i, deprel in enumerate(sentence.deprels, start=1):
        karaka = _extract_karaka(deprel)
        if karaka is None:
            continue
        role = karaka_to_propbank.get(karaka)
        if role is None:
            continue

        span = _subtree_indices(i, children)
        for position, token_idx in enumerate(span):
            prefix = "B" if position == 0 else "I"
            tags[token_idx - 1] = f"{prefix}-{role}"

    return tags


def find_predicate_index(sentence: ConllSentence) -> Optional[int]:
    """Return the 0-indexed position of the sentence's root token (the predicate).

    The Paninian Karaka relations are all defined relative to the governing
    verb, which is the dependency root (``HEAD == 0``) in these simple,
    single-clause sentences.
    """
    for i, head in enumerate(sentence.heads):
        if head == 0:
            return i
    return None


def build_examples_from_conllu(
    text: str,
    morphology_extractor: Optional[MorphologyExtractor] = None,
    karaka_to_propbank: Optional[Dict[str, str]] = None,
) -> List[SRLExample]:
    """Parse CoNLL-U text into ready-to-train ``SRLExample`` objects.

    Sentences with no dependency root (and therefore no identifiable
    predicate) are skipped. A sentence's optional
    ``# predicate_synset_id = <int>`` comment (see :func:`parse_conllu`)
    supplies its gold IndoWordNet synset id for WSD supervision, if present.
    """
    morphology_extractor = morphology_extractor or MorphologyExtractor()
    karaka_to_propbank = karaka_to_propbank or load_karaka_to_propbank()

    examples: List[SRLExample] = []
    for sentence in parse_conllu(text):
        predicate_index = find_predicate_index(sentence)
        if predicate_index is None:
            continue

        bio_tags = sentence_to_bio(sentence, karaka_to_propbank)
        morph = morphology_extractor.extract(sentence.tokens)

        examples.append(
            SRLExample(
                tokens=sentence.tokens,
                bio_tags=bio_tags,
                predicate_index=predicate_index,
                case_marker_ids=[m.case_marker_id for m in morph],
                clitic_ids=[m.clitic_id for m in morph],
                lemmas=sentence.lemmas,
                predicate_lemma=sentence.lemmas[predicate_index],
                predicate_synset_id=sentence.predicate_synset_id,
            )
        )

    return examples
