"""Traces the neural encoder + role scorer to ONNX and applies INT8 dynamic quantization.

Per Section 6 (Phase 6) of context.txt: only the IndicBERT/MuRIL feature
extractor, BiLSTM, and role-scorer MLP are exported. The ontology masking
matrix generation and the constrained Viterbi search stay in native
Python/NumPy (``src/ontology/mask_generator.py``,
``src/inference/viterbi_decoder.py``) since dynamic graph search with
arbitrary lexicon lookups does not belong in a static ONNX graph. The WSD
head is likewise kept in PyTorch, run separately against the exported
graph's pooled predicate-hidden-state output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.models.encoder import SRLEncoder
from src.models.role_scorer import RoleScorer, pool_predicate_hidden_state


class ExportableSRLNet(nn.Module):
    """Encoder (backbone + morphology embeddings + BiLSTM) + role scorer, ONNX-traceable.

    Runs the BiLSTM directly over the padded batch instead of going through
    ``pack_padded_sequence``/``pad_packed_sequence`` (as ``SRLEncoder`` does
    for training efficiency): RNN packing ops are not reliably traceable to
    ONNX, and since an LSTM is strictly causal per batch row, padding tokens
    *after* a sequence's true length never influence any earlier, valid
    position's hidden state — so every position that downstream code
    actually reads back out is numerically identical either way.
    """

    def __init__(self, encoder: SRLEncoder, role_scorer: RoleScorer) -> None:
        super().__init__()
        self.backbone = encoder.backbone
        self.case_embedding = encoder.case_embedding
        self.clitic_embedding = encoder.clitic_embedding
        self.predicate_embedding = encoder.predicate_embedding
        self.bilstm = encoder.bilstm
        self.role_scorer = role_scorer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        case_marker_ids: torch.Tensor,
        clitic_ids: torch.Tensor,
        predicate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone_out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        composite = torch.cat(
            [
                backbone_out,
                self.case_embedding(case_marker_ids),
                self.clitic_embedding(clitic_ids),
                self.predicate_embedding(predicate_mask),
            ],
            dim=-1,
        )
        hidden_states, _ = self.bilstm(composite)
        emissions = self.role_scorer(hidden_states, predicate_mask)
        predicate_hidden = pool_predicate_hidden_state(hidden_states, predicate_mask)
        return emissions, predicate_hidden


_INPUT_NAMES = [
    "input_ids",
    "attention_mask",
    "case_marker_ids",
    "clitic_ids",
    "predicate_mask",
]
_OUTPUT_NAMES = ["emissions", "predicate_hidden"]
_DYNAMIC_AXES = {
    **{name: {0: "batch", 1: "seq_len"} for name in _INPUT_NAMES},
    "emissions": {0: "batch", 1: "seq_len"},
    "predicate_hidden": {0: "batch"},
}


def export_to_onnx(
    encoder: SRLEncoder,
    role_scorer: RoleScorer,
    output_path: Path,
    seq_len: int = 16,
    opset_version: int = 17,
) -> Path:
    """Trace ``ExportableSRLNet`` and write it to ``output_path`` (FP32 ONNX)."""
    model = ExportableSRLNet(encoder, role_scorer)
    model.eval()

    dummy_input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    dummy_case_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_clitic_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_predicate_mask = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_predicate_mask[0, 0] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                dummy_input_ids,
                dummy_attention_mask,
                dummy_case_ids,
                dummy_clitic_ids,
                dummy_predicate_mask,
            ),
            str(output_path),
            input_names=_INPUT_NAMES,
            output_names=_OUTPUT_NAMES,
            dynamic_axes=_DYNAMIC_AXES,
            opset_version=opset_version,
            dynamo=False,  # legacy TorchScript-based exporter; avoids the
            # torch 2.x default dynamo path's onnxscript dependency, which
            # isn't in this project's requirements.
        )
    return output_path


def quantize_to_int8(onnx_path: Path, quantized_path: Path) -> Path:
    """Apply ``onnxruntime.quantization.quantize_dynamic`` targeting QInt8 weights."""
    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )
    return quantized_path
