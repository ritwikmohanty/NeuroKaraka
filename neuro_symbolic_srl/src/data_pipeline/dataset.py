"""PyTorch ``Dataset`` and dynamic-padding collator for the SRL model.

Bridges word-level annotations (BIO PropBank tags, predicate index,
morphological feature IDs from :mod:`morphology_extractor`) to the subword
tokenization produced by the Indic transformer backbone, using the fast
tokenizer's ``word_ids()`` alignment. Word-level features are propagated to
every subword of their word; word-level BIO tags follow the standard
subword-continuation convention (``B-X`` on the first subword, ``I-X`` on
continuation subwords of the same word).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"
)
LABEL_PAD_ID = -100


def _load_default_backbone_and_max_len() -> tuple[str, int]:
    with open(_MODEL_CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    encoder_cfg = config["encoder"]
    return encoder_cfg["backbone_name"], encoder_cfg["max_seq_len"]


@dataclass
class SRLExample:
    """A single word-level annotated sentence for one predicate frame.

    ``lemmas``/``predicate_lemma``/``predicate_synset_id`` are optional
    because the tokenization/collation path below never needs them — only
    ``lemmas`` for IndoWordNet lookups (which are indexed by dictionary
    form, not inflected surface tokens) and WSD gold supervision, both
    consumed directly by ``scripts/train.py`` and ``src/inference/pipeline.py``.
    """

    tokens: List[str]
    bio_tags: List[str]
    predicate_index: int
    case_marker_ids: List[int]
    clitic_ids: List[int]
    lemmas: Optional[List[str]] = None
    predicate_lemma: Optional[str] = None
    predicate_synset_id: Optional[int] = None


def _propagate_bio_to_subwords(
    word_ids: List[Optional[int]], bio_tags: List[str]
) -> List[str]:
    """Expand word-level BIO tags to subword-level tags.

    The first subword of a word keeps the word's original tag (``B-X``,
    ``I-X``, or ``O``); continuation subwords of a ``B-X`` word become
    ``I-X`` so the sequence stays a well-formed BIO span. Special tokens
    (``word_id is None``) are handled separately by the caller.
    """
    tags: List[str] = []
    previous_word_id: Optional[int] = None
    for word_id in word_ids:
        if word_id is None:
            tags.append("O")
        else:
            tag = bio_tags[word_id]
            is_continuation_subword = word_id == previous_word_id
            if is_continuation_subword and tag.startswith("B-"):
                tags.append("I-" + tag[2:])
            else:
                tags.append(tag)
        previous_word_id = word_id
    return tags


class SRLDataset(Dataset):
    """Tokenizes :class:`SRLExample` items and aligns per-word features to subwords."""

    def __init__(
        self,
        examples: List[SRLExample],
        label_to_id: Dict[str, int],
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        default_backbone, default_max_len = _load_default_backbone_and_max_len()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(default_backbone)
        if not self.tokenizer.is_fast:
            raise ValueError(
                "SRLDataset requires a fast tokenizer (word_ids() support); "
                f"got a slow tokenizer for {self.tokenizer.name_or_path!r}."
            )
        self.max_seq_len = max_seq_len or default_max_len
        self.examples = examples
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List]:
        example = self.examples[index]
        encoding = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        word_ids = encoding.word_ids()

        subword_tags = _propagate_bio_to_subwords(word_ids, example.bio_tags)
        label_ids = [
            LABEL_PAD_ID if wid is None else self.label_to_id[tag]
            for wid, tag in zip(word_ids, subword_tags)
        ]

        case_marker_ids = [
            0 if wid is None else example.case_marker_ids[wid] for wid in word_ids
        ]
        clitic_ids = [
            0 if wid is None else example.clitic_ids[wid] for wid in word_ids
        ]
        predicate_mask = [
            1 if wid == example.predicate_index else 0 for wid in word_ids
        ]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "word_ids": word_ids,
            "case_marker_ids": case_marker_ids,
            "clitic_ids": clitic_ids,
            "predicate_mask": predicate_mask,
            "label_ids": label_ids,
        }


class SRLCollator:
    """Dynamic-padding collate function producing batched ``torch.Tensor``s."""

    _TENSOR_FIELDS_AND_PAD_VALUES = {
        "input_ids": None,  # filled in from tokenizer.pad_token_id
        "attention_mask": 0,
        "case_marker_ids": 0,
        "clitic_ids": 0,
        "predicate_mask": 0,
        "label_ids": LABEL_PAD_ID,
    }

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, label_pad_id: int = LABEL_PAD_ID
    ) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, List]]) -> Dict[str, object]:
        max_len = max(len(item["input_ids"]) for item in batch)

        pad_values = dict(self._TENSOR_FIELDS_AND_PAD_VALUES)
        pad_values["input_ids"] = self.pad_token_id
        pad_values["label_ids"] = self.label_pad_id

        collated: Dict[str, object] = {}
        for field, pad_value in pad_values.items():
            padded_rows = []
            for item in batch:
                row = list(item[field])
                row.extend([pad_value] * (max_len - len(row)))
                padded_rows.append(row)
            collated[field] = torch.tensor(padded_rows, dtype=torch.long)

        collated["word_ids"] = [item["word_ids"] for item in batch]
        return collated
