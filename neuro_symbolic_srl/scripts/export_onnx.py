"""Export a trained checkpoint to ONNX and quantize it to INT8.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint checkpoints/final_model.pt --output-dir checkpoints

Produces ``srl_net_int8.onnx`` in the output directory (the file the API
loads). The intermediate FP32 graph (~1 GB) is deleted afterwards unless
``--keep-fp32`` is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference.onnx_exporter import export_to_onnx, quantize_to_int8
from src.models.encoder import SRLEncoder, load_model_config
from src.models.role_scorer import RoleScorer

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=str(_REPO_ROOT / "checkpoints" / "final_model.pt")
    )
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "checkpoints"))
    parser.add_argument("--keep-fp32", action="store_true")
    args = parser.parse_args()

    model_config = load_model_config()
    num_labels = len(model_config["crf"]["label_set"])

    encoder = SRLEncoder(model_config)
    role_scorer = RoleScorer(
        hidden_dim=encoder.output_dim,
        mlp_hidden_dim=model_config["role_scorer"]["mlp_hidden_dim"],
        num_labels=num_labels,
        dropout=model_config["role_scorer"]["dropout"],
    )

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(
        {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    )
    role_scorer.load_state_dict(
        {k[len("role_scorer."):]: v for k, v in state.items() if k.startswith("role_scorer.")}
    )

    output_dir = Path(args.output_dir)
    fp32_path = export_to_onnx(encoder, role_scorer, output_dir / "srl_net.onnx")
    int8_path = quantize_to_int8(fp32_path, output_dir / "srl_net_int8.onnx")

    print(f"FP32 ONNX: {fp32_path.stat().st_size / 1e6:.0f} MB")
    print(f"INT8 ONNX: {int8_path.stat().st_size / 1e6:.0f} MB -> {int8_path}")

    if not args.keep_fp32:
        fp32_path.unlink()
        print("Removed intermediate FP32 graph (pass --keep-fp32 to retain it).")


if __name__ == "__main__":
    main()
