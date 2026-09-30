"""Production FastAPI application serving the Neuro-Symbolic SRL engine.

Loads the quantized ONNX inference session (encoder + role scorer) and the
IndoWordNet lexicon once at startup via a lifespan context manager. The WSD
head and CRF transition parameters are small enough to stay in PyTorch;
per Section 6 Phase 6 of context.txt, the ontology masking matrix generation
and constrained Viterbi search also stay in native Python (not part of the
ONNX graph).

Known limitation: this service has no morphological lemmatizer or
dependency parser. ``POST /api/v1/parse_frame`` whitespace-tokenizes raw
text and uses the surface tokens themselves as a lemma proxy (many short
Hindi nominals are already close to their dictionary form, but inflected
verbs are not — e.g. "खाया" != its lemma "खाना" — so IndoWordNet-grounded
lookups for the predicate itself are less reliable here than in the
CoNLL-U-fed training/testing path, which always supplies gold lemmas).
Predicate auto-detection (when ``target_predicate`` is omitted) is a simple
"last token" SOV-order heuristic, not real verb identification.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import pyiwn
import torch
import yaml
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from api.schemas import Argument, ParseFrameRequest, ParseFrameResponse, Predicate
from src.data_pipeline.lemmatizer import guess_noun_lemma, guess_verb_lemma
from src.data_pipeline.morphology_extractor import MorphologyExtractor
from src.inference.viterbi_decoder import decode, subword_labels_to_word_labels
from src.models.crf_layer import LinearChainCRF
from src.models.wsd_head import WSDHead
from src.ontology.gloss_cache import GlossEmbeddingCache
from src.ontology.hypernym_engine import HypernymEngine
from src.ontology.indowordnet_client import IndoWordNetClient
from src.ontology.mask_generator import MaskGenerator

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CONFIG_PATH = _REPO_ROOT / "config" / "model_config.yaml"
_ONTOLOGY_RULES_PATH = _REPO_ROOT / "config" / "ontology_rules.yaml"

ONNX_MODEL_PATH = Path(
    os.environ.get("SRL_ONNX_MODEL_PATH", _REPO_ROOT / "checkpoints" / "srl_net_int8.onnx")
)
CHECKPOINT_PATH = Path(
    os.environ.get("SRL_CHECKPOINT_PATH", _REPO_ROOT / "checkpoints" / "final_model.pt")
)


class SRLServiceState:
    """Bundles every loaded component the endpoint needs, built once at startup."""

    def __init__(self) -> None:
        with open(_MODEL_CONFIG_PATH, encoding="utf-8") as fh:
            self.model_config = yaml.safe_load(fh)
        with open(_ONTOLOGY_RULES_PATH, encoding="utf-8") as fh:
            karaka_to_propbank = yaml.safe_load(fh)["karaka_to_propbank"]
        self.propbank_to_karaka: Dict[str, str] = {
            role: karaka for karaka, role in karaka_to_propbank.items()
        }

        self.label_list: List[str] = self.model_config["crf"]["label_set"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["encoder"]["backbone_name"]
        )
        self.morphology_extractor = MorphologyExtractor()

        self.iwn_client = IndoWordNetClient()
        self.hypernym_engine = HypernymEngine(client=self.iwn_client)
        self.mask_generator = MaskGenerator(hypernym_engine=self.hypernym_engine)
        self.gloss_cache = GlossEmbeddingCache(iwn_client=self.iwn_client)

        self.onnx_session = ort.InferenceSession(
            str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
        )

        self.crf = LinearChainCRF(num_labels=len(self.label_list))
        self.wsd_head = WSDHead(
            hidden_dim=self.model_config["bilstm"]["hidden_dim"] * 2,
            gloss_dim=self.gloss_cache.embedding_dim,
            projection_dim=self.model_config["wsd_head"]["projection_dim"],
        )
        if CHECKPOINT_PATH.exists():
            state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
            self.crf.load_state_dict(
                {k[len("crf."):]: v for k, v in state_dict.items() if k.startswith("crf.")}
            )
            self.wsd_head.load_state_dict(
                {
                    k[len("wsd_head."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("wsd_head.")
                }
            )
        self.crf.eval()
        self.wsd_head.eval()

    def parse_frame(self, text: str, target_predicate: Optional[int]) -> ParseFrameResponse:
        tokens = text.strip().split()
        if not tokens:
            raise HTTPException(status_code=422, detail="text must contain at least one token")

        predicate_index = target_predicate if target_predicate is not None else len(tokens) - 1
        if not (0 <= predicate_index < len(tokens)):
            raise HTTPException(
                status_code=422,
                detail=f"target_predicate {predicate_index} out of range for {len(tokens)} tokens",
            )

        # No dependency parser or full lemmatizer is available (see module
        # docstring), so lemmas are dictionary-validated best-effort guesses
        # (src/data_pipeline/lemmatizer.py) rather than true lemmatization:
        # the predicate gets a verb-inflection guess (mask_generator's
        # animacy gate is keyed by verb lemma, e.g. "खोलना"), every other
        # token gets a noun oblique-case guess (animacy/location/time checks
        # are keyed by noun lemma, e.g. "बच्चा" not its oblique "बच्चे").
        # Skipping either silently defeats the ontology mask's guarantees
        # for exactly the tokens it's meant to protect.
        lemmas = [
            guess_noun_lemma(token, client=self.iwn_client) for token in tokens
        ]
        lemmas[predicate_index] = guess_verb_lemma(
            tokens[predicate_index], client=self.iwn_client
        )
        predicate_lemma = lemmas[predicate_index]

        morph = self.morphology_extractor.extract(tokens)
        case_markers = [m.case_marker for m in morph]
        word_case_ids = [m.case_marker_id for m in morph]
        word_clitic_ids = [m.clitic_id for m in morph]

        encoding = self.tokenizer(tokens, is_split_into_words=True)
        word_ids = encoding.word_ids()

        def _expand(word_level: List[int]) -> np.ndarray:
            return np.array(
                [[0 if wid is None else word_level[wid] for wid in word_ids]], dtype=np.int64
            )

        input_ids = np.array([encoding["input_ids"]], dtype=np.int64)
        attention_mask = np.array([encoding["attention_mask"]], dtype=np.int64)
        case_marker_ids = _expand(word_case_ids)
        clitic_ids = _expand(word_clitic_ids)
        predicate_mask = np.array(
            [[1 if wid == predicate_index else 0 for wid in word_ids]], dtype=np.int64
        )

        onnx_emissions, onnx_predicate_hidden = self.onnx_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "case_marker_ids": case_marker_ids,
                "clitic_ids": clitic_ids,
                "predicate_mask": predicate_mask,
            },
        )
        neural_emissions = torch.from_numpy(onnx_emissions)
        predicate_hidden = torch.from_numpy(onnx_predicate_hidden)

        sense_info = self._disambiguate_sense(predicate_lemma, predicate_hidden)

        word_level_mask = self.mask_generator.generate(lemmas, predicate_lemma, case_markers)
        subword_mask = torch.zeros(1, len(word_ids), len(self.label_list))
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
            mask=torch.from_numpy(attention_mask[0]),
        )
        word_labels = subword_labels_to_word_labels(subword_labels, word_ids, len(tokens))

        return ParseFrameResponse(
            predicate=Predicate(
                text=tokens[predicate_index], lemma=predicate_lemma, **sense_info
            ),
            arguments=self._extract_arguments(tokens, word_labels),
        )

    def _disambiguate_sense(
        self, predicate_lemma: str, predicate_hidden: torch.Tensor
    ) -> Dict[str, object]:
        max_candidates = self.model_config["wsd_head"]["max_candidate_senses"]
        candidates, gloss_embeddings = self.gloss_cache.embed_candidates(
            predicate_lemma, pyiwn.PosTag.VERB, max_candidates=max_candidates
        )
        if not candidates:
            return {"sense": None, "synset_id": None}

        with torch.no_grad():
            alpha, _ = self.wsd_head(
                predicate_hidden,
                gloss_embeddings.unsqueeze(0),
                torch.ones(1, len(candidates)),
            )
        best_index = int(torch.argmax(alpha, dim=-1).item())
        best_synset = candidates[best_index]
        return {"sense": best_synset.gloss(), "synset_id": best_synset.synset_id()}

    def _extract_arguments(self, tokens: List[str], word_labels: List[str]) -> List[Argument]:
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
                    Argument(
                        span=[start, j - 1],
                        text=" ".join(tokens[start:j]),
                        karaka=self.propbank_to_karaka.get(role),
                        role=role,
                    )
                )
                i = j
            else:
                i += 1
        return arguments


_state: Optional[SRLServiceState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = SRLServiceState()
    yield
    _state = None


app = FastAPI(
    title="Neuro-Symbolic SRL Engine",
    description=(
        "Semantic Role Labeling for Hindi, combining a BiLSTM-over-MuRIL "
        "neural scorer with IndoWordNet ontology masking and a constrained "
        "Viterbi-CRF decoder."
    ),
    lifespan=lifespan,
)


@app.post("/api/v1/parse_frame", response_model=ParseFrameResponse)
def parse_frame(request: ParseFrameRequest) -> ParseFrameResponse:
    if _state is None:
        raise HTTPException(status_code=503, detail="Service is still starting up")
    return _state.parse_frame(request.text, request.target_predicate)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok" if _state is not None else "starting"}
