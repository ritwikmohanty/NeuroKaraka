"""Integration tests for ONNX export + INT8 dynamic quantization.

These are necessarily heavier than the rest of the suite (they trace and
quantize the real MuRIL-backed encoder), so everything is bundled behind one
module-scoped fixture pair rather than re-exporting per test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.inference.onnx_exporter import ExportableSRLNet, export_to_onnx, quantize_to_int8
from src.models.encoder import SRLEncoder, load_model_config
from src.models.role_scorer import RoleScorer

_SEQ_LEN = 8


@pytest.fixture(scope="module")
def model_config():
    return load_model_config()


@pytest.fixture(scope="module")
def encoder_and_role_scorer(model_config):
    encoder = SRLEncoder(model_config)
    role_scorer = RoleScorer(
        hidden_dim=encoder.output_dim,
        mlp_hidden_dim=model_config["role_scorer"]["mlp_hidden_dim"],
        num_labels=len(model_config["crf"]["label_set"]),
        dropout=model_config["role_scorer"]["dropout"],
    )
    return encoder, role_scorer


@pytest.fixture(scope="module")
def net(encoder_and_role_scorer) -> ExportableSRLNet:
    encoder, role_scorer = encoder_and_role_scorer
    module = ExportableSRLNet(encoder, role_scorer)
    module.eval()
    return module


@pytest.fixture(scope="module")
def exported_paths(tmp_path_factory, encoder_and_role_scorer):
    encoder, role_scorer = encoder_and_role_scorer
    export_dir = tmp_path_factory.mktemp("onnx_export")
    fp32_path = export_to_onnx(encoder, role_scorer, export_dir / "net.onnx", seq_len=_SEQ_LEN)
    int8_path = quantize_to_int8(fp32_path, export_dir / "net_int8.onnx")
    return fp32_path, int8_path


def _dummy_inputs(seq_len: int = _SEQ_LEN):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    case_marker_ids = torch.zeros(1, seq_len, dtype=torch.long)
    clitic_ids = torch.zeros(1, seq_len, dtype=torch.long)
    predicate_mask = torch.zeros(1, seq_len, dtype=torch.long)
    predicate_mask[0, 1] = 1
    return input_ids, attention_mask, case_marker_ids, clitic_ids, predicate_mask


class TestExportToOnnx:
    def test_produces_a_loadable_onnx_file(self, exported_paths) -> None:
        fp32_path, _ = exported_paths
        assert fp32_path.exists()
        session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        assert {o.name for o in session.get_outputs()} == {"emissions", "predicate_hidden"}

    def test_fp32_onnx_output_matches_pytorch_closely(
        self, net: ExportableSRLNet, exported_paths, model_config
    ) -> None:
        fp32_path, _ = exported_paths
        inputs = _dummy_inputs()
        with torch.no_grad():
            torch_emissions, torch_pred_hidden = net(*inputs)

        session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        input_names = ["input_ids", "attention_mask", "case_marker_ids", "clitic_ids", "predicate_mask"]
        onnx_emissions, onnx_pred_hidden = session.run(
            None, {name: tensor.numpy() for name, tensor in zip(input_names, inputs)}
        )

        assert np.allclose(onnx_emissions, torch_emissions.numpy(), atol=1e-3)
        assert np.allclose(onnx_pred_hidden, torch_pred_hidden.numpy(), atol=1e-3)

    def test_output_shapes_match_num_labels_and_hidden_dim(
        self, exported_paths, model_config
    ) -> None:
        fp32_path, _ = exported_paths
        inputs = _dummy_inputs()
        session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        input_names = ["input_ids", "attention_mask", "case_marker_ids", "clitic_ids", "predicate_mask"]
        emissions, pred_hidden = session.run(
            None, {name: tensor.numpy() for name, tensor in zip(input_names, inputs)}
        )
        num_labels = len(model_config["crf"]["label_set"])
        hidden_dim = model_config["bilstm"]["hidden_dim"] * 2
        assert emissions.shape == (1, _SEQ_LEN, num_labels)
        assert pred_hidden.shape == (1, hidden_dim)


class TestQuantizeToInt8:
    def test_quantized_file_is_smaller(self, exported_paths) -> None:
        fp32_path, int8_path = exported_paths
        assert int8_path.exists()
        assert int8_path.stat().st_size < fp32_path.stat().st_size

    def test_quantized_output_is_finite_and_same_shape_as_fp32(
        self, exported_paths, model_config
    ) -> None:
        fp32_path, int8_path = exported_paths
        inputs = _dummy_inputs()
        input_names = ["input_ids", "attention_mask", "case_marker_ids", "clitic_ids", "predicate_mask"]
        feed = {name: tensor.numpy() for name, tensor in zip(input_names, inputs)}

        fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        fp32_emissions, _ = fp32_session.run(None, feed)
        int8_emissions, _ = int8_session.run(None, feed)

        assert int8_emissions.shape == fp32_emissions.shape
        assert np.isfinite(int8_emissions).all()
