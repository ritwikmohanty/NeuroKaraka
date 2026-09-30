"""IndoWordNet gloss embedding cache for the WSD head (Section 5 Phase 1 of context.txt).

Precomputes and caches dense sentence-embedding vectors for candidate
synsets' glosses (+ example sentences), so repeated lookups during training
and inference don't repeatedly re-embed the same text. Cached to a single
``torch.save``-serialized file under ``data/indowordnet_cache/``, keyed by
IndoWordNet synset id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyiwn
import torch
import yaml
from sentence_transformers import SentenceTransformer

from src.ontology.indowordnet_client import IndoWordNetClient

_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"
)
_DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "indowordnet_cache"
    / "gloss_embeddings.pt"
)


class GlossEmbeddingCache:
    """Embeds and caches IndoWordNet synset glosses for the WSD head."""

    def __init__(
        self,
        iwn_client: Optional[IndoWordNetClient] = None,
        model_name: Optional[str] = None,
        cache_path: Path = _DEFAULT_CACHE_PATH,
    ) -> None:
        self._client = iwn_client or IndoWordNetClient()

        if model_name is None:
            with open(_MODEL_CONFIG_PATH, encoding="utf-8") as fh:
                model_name = yaml.safe_load(fh)["wsd_head"]["gloss_encoder_name"]

        self._encoder = SentenceTransformer(model_name)
        self.embedding_dim = self._encoder.get_embedding_dimension()

        self._cache_path = cache_path
        self._cache: Dict[int, torch.Tensor] = {}
        self._load()

    def _load(self) -> None:
        if self._cache_path.exists():
            self._cache = torch.load(self._cache_path, weights_only=True)

    def save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._cache, self._cache_path)

    def _gloss_text(self, synset: "pyiwn.iwn.Synset") -> str:
        text = synset.gloss()
        examples = synset.examples()
        if examples:
            text = text + " " + " ".join(examples)
        return text

    def embed_synset(self, synset: "pyiwn.iwn.Synset") -> torch.Tensor:
        synset_id = synset.synset_id()
        cached = self._cache.get(synset_id)
        if cached is not None:
            return cached

        embedding = self._encoder.encode(
            self._gloss_text(synset), convert_to_tensor=True
        )
        self._cache[synset_id] = embedding
        return embedding

    def candidates(
        self,
        lemma: str,
        pos: Optional["pyiwn.PosTag"] = None,
        max_candidates: Optional[int] = None,
    ) -> List["pyiwn.iwn.Synset"]:
        synsets = list(self._client.synsets(lemma, pos))
        if max_candidates is not None:
            synsets = synsets[:max_candidates]
        return synsets

    def embed_candidates(
        self,
        lemma: str,
        pos: Optional["pyiwn.PosTag"] = None,
        max_candidates: Optional[int] = None,
    ) -> Tuple[List["pyiwn.iwn.Synset"], torch.Tensor]:
        """Return (candidate synsets, stacked gloss embeddings ``(K, embedding_dim)``)."""
        synsets = self.candidates(lemma, pos, max_candidates)
        if not synsets:
            return [], torch.zeros(0, self.embedding_dim)
        embeddings = torch.stack([self.embed_synset(s) for s in synsets])
        return synsets, embeddings
