"""Benchmarks p50/p95/p99 CPU inference latency of the quantized ONNX SRL model.

Measures only the exported neural graph (encoder + role scorer) via
onnxruntime — the piece Section 7 of context.txt's latency target (p95 <=
20ms) is actually about — not the full pipeline's Python-side ontology
masking / Viterbi decode, which are comparatively cheap, or the WSD head's
gloss-embedding lookups, which depend on IndoWordNet I/O rather than model
inference.

Usage:
    python scripts/benchmark_latency.py --onnx-model checkpoints/srl_net_int8.onnx
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CONFIG_PATH = _REPO_ROOT / "config" / "model_config.yaml"

_BENCHMARK_TOKENS = ["पुलिस", "ने", "दूरबीन", "से", "चोर", "को", "देखा"]
_PREDICATE_WORD_INDEX = 6


def _build_inputs(tokenizer, tokens: List[str], predicate_word_index: int) -> Dict[str, np.ndarray]:
    encoding = tokenizer(tokens, is_split_into_words=True)
    word_ids = encoding.word_ids()
    seq_len = len(encoding["input_ids"])
    return {
        "input_ids": np.array([encoding["input_ids"]], dtype=np.int64),
        "attention_mask": np.array([encoding["attention_mask"]], dtype=np.int64),
        "case_marker_ids": np.zeros((1, seq_len), dtype=np.int64),
        "clitic_ids": np.zeros((1, seq_len), dtype=np.int64),
        "predicate_mask": np.array(
            [[1 if wid == predicate_word_index else 0 for wid in word_ids]], dtype=np.int64
        ),
    }


def benchmark(
    onnx_path: Path,
    num_requests: int = 1000,
    num_warmup: int = 20,
    intra_op_threads: int = 1,
) -> Dict[str, float]:
    with open(_MODEL_CONFIG_PATH, encoding="utf-8") as fh:
        model_config = yaml.safe_load(fh)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(
        str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config["encoder"]["backbone_name"])
    feed = _build_inputs(tokenizer, _BENCHMARK_TOKENS, _PREDICATE_WORD_INDEX)

    for _ in range(num_warmup):
        session.run(None, feed)

    latencies_ms = np.empty(num_requests, dtype=np.float64)
    for i in range(num_requests):
        start = time.perf_counter()
        session.run(None, feed)
        latencies_ms[i] = (time.perf_counter() - start) * 1000.0

    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    return {
        "num_requests": num_requests,
        "mean_ms": float(latencies_ms.mean()),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "min_ms": float(latencies_ms.min()),
        "max_ms": float(latencies_ms.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-model",
        default=str(_REPO_ROOT / "checkpoints" / "srl_net_int8.onnx"),
        help="Path to the (quantized) ONNX model to benchmark.",
    )
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--intra-op-threads", type=int, default=1)
    parser.add_argument(
        "--p95-target-ms",
        type=float,
        default=20.0,
        help="Report pass/fail against this p95 latency target (ms).",
    )
    args = parser.parse_args()

    results = benchmark(
        Path(args.onnx_model), args.num_requests, args.num_warmup, args.intra_op_threads
    )

    print(f"Requests:  {results['num_requests']}")
    print(f"Mean:      {results['mean_ms']:.2f} ms")
    print(f"p50:       {results['p50_ms']:.2f} ms")
    print(f"p95:       {results['p95_ms']:.2f} ms")
    print(f"p99:       {results['p99_ms']:.2f} ms")
    print(f"Min / Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")

    if results["p95_ms"] <= args.p95_target_ms:
        print(f"PASS: p95 latency {results['p95_ms']:.2f}ms <= {args.p95_target_ms}ms target")
    else:
        print(f"FAIL: p95 latency {results['p95_ms']:.2f}ms exceeds {args.p95_target_ms}ms target")


if __name__ == "__main__":
    main()
