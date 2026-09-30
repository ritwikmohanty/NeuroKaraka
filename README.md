# NeuroKaraka

A Neuro-Symbolic Semantic Role Labeling (SRL) engine for Hindi. A BiLSTM-over-MuRIL
neural encoder proposes argument-role emissions; an **IndoWordNet-grounded ontology
mask** forbids semantically impossible assignments (e.g. an inanimate noun as the
volitional agent of an intentional verb) before a constrained **Viterbi-CRF** decodes
the globally best, linguistically valid parse. Full architectural spec: [`context.txt`](context.txt).

All code lives under [`neuro_symbolic_srl/`](neuro_symbolic_srl/) — that's the
project root for every command below. Setting this up on a new machine (incl.
macOS) or presenting it? See **[DEMO_GUIDE.md](DEMO_GUIDE.md)**: full setup,
live-demo steps, a timed presentation script, and Q&A prep.

```
चाबी ने ताला खोला              → चाबी = ARGM-INS (not ARG0 — it's the instrument, not the agent)
सेब को बच्चे ने खाया            → सेब = ARG1, बच्चे = ARG0 (scrambled OSV, resolved correctly)
पुलिस ने दूरबीन से चोर को देखा  → दूरबीन = ARGM-INS, attached to the verb, not चोर
```

## Repository layout

```
neuro_symbolic_srl/
├── config/               model/training hyperparameters, Karaka<->PropBank + ontology rules
├── data/
│   ├── raw/              CoNLL-U corpora (toy_train.conllu / toy_val.conllu — see Limitations)
│   ├── processed/        (reserved; unused — the pipeline reads CoNLL-U directly)
│   └── indowordnet_cache/ cached IndoWordNet gloss embeddings (created on first WSD use, reused after)
├── src/
│   ├── data_pipeline/    CoNLL-U parsing, morphology extraction, lemma guessers, PyTorch Dataset
│   ├── ontology/         IndoWordNet client, hypernym/selectional-restriction engine, mask generator
│   ├── models/           encoder, WSD head, role scorer, CRF layer
│   ├── inference/        Viterbi decoder, ONNX exporter, end-to-end pipeline
│   └── utils/            (reserved; unused)
├── api/                  FastAPI app (main.py) + Pydantic schemas (schemas.py)
├── scripts/              train.py, export_onnx.py, demo.py, benchmark_latency.py
├── tests/                pytest suite (130 tests across every module above)
├── checkpoints/          trained weights + exported ONNX models land here (git-ignored)
└── pyproject.toml
```

## Prerequisites

- **Python 3.10+** (developed/tested on 3.12)
- **~8 GB free disk**: measured cache sizes are ~1.8 GB (MuRIL backbone),
  ~1.1 GB (the multilingual sentence-transformer used for WSD gloss
  embeddings), ~140 MB (IndoWordNet data) — plus your own checkpoints
  (`checkpoints/final_model.pt` is ~1 GB, the FP32 ONNX export another ~1 GB,
  the quantized INT8 export ~260 MB)
- **Internet access on first run** — several resources auto-download the first
  time they're used (see below); after that, everything is cached locally and
  works offline
- Windows, macOS, or Linux — no OS-specific code paths except one Windows
  console-encoding workaround, which is already handled for you (see below)

## Setup on a new machine

```bash
cd neuro_symbolic_srl
python -m venv .venv

# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# macOS / Linux:          source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"   # add [dev] to also get pytest; drop it for a leaner runtime-only install
```

`pyproject.toml` declares every dependency (torch, transformers, pyiwn, seqeval,
fastapi, uvicorn, pydantic, onnx, onnxruntime, sentence-transformers, pyyaml,
pytest). `torch` will pull whatever wheel `pip` resolves for your platform; on a
CPU-only machine you can instead run
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first if you
want to avoid a CUDA-enabled download.

### First run: what downloads automatically, and where it's cached

| Resource | Trigger | Size on disk | Cached at |
|---|---|---|---|
| IndoWordNet Hindi data | first `import pyiwn` (anything touching `src/ontology/`) | ~140 MB | `~/iwn_data` |
| `google/muril-base-cased` (tokenizer + backbone) | first use of the encoder/pipeline/API | ~1.8 GB | `~/.cache/huggingface/hub` |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | first WSD/gloss-embedding use | ~1.1 GB | `~/.cache/huggingface/hub` |

Nothing needs to be downloaded manually — just run any command below and let it
happen once. Set `HF_HUB_OFFLINE=1` afterward to skip the Hugging Face Hub's
redundant "is this cached file still current" network checks and speed things up.

### Windows note

`pyiwn`'s bundled data files are UTF-8, but Windows' default console/file
encoding usually isn't — this is already worked around inside
[`src/ontology/indowordnet_client.py`](neuro_symbolic_srl/src/ontology/indowordnet_client.py),
so you don't need to do anything for normal use. For verbose `pytest` output
with Devanagari text in failure messages, it's still worth running with UTF-8
I/O explicitly:

```bash
# Git Bash / WSL
PYTHONIOENCODING=utf-8 pytest tests/ -q

# PowerShell
$env:PYTHONIOENCODING = "utf-8"; pytest tests/ -q
```

## Running the tests

```bash
pytest tests/ -q            # full suite (~130 tests, ~2-3 min incl. model downloads/first-run caching)
pytest tests/test_ontology.py -q     # IndoWordNet client, hypernym engine, mask generator
pytest tests/test_morphology.py -q   # case/clitic extraction, CoNLL-U parsing, lemma guessers, Dataset
pytest tests/test_crf.py -q          # CRF, Viterbi decoder, encoder/WSD-head/role-scorer
pytest tests/test_onnx_export.py -q  # ONNX export + INT8 quantization (heaviest single file, ~1-2 min)
pytest tests/test_pipeline.py -q     # end-to-end verification sentences (needs a trained checkpoint)
pytest tests/test_api.py -q          # FastAPI endpoint (needs an exported ONNX model)
```

`test_pipeline.py` and `test_api.py` skip their checkpoint-dependent assertions
gracefully if `checkpoints/final_model.pt` / `checkpoints/srl_net_int8.onnx`
don't exist yet — run **Training** and **Exporting to ONNX** below first to get
full coverage.

## Training

There is no bundled copy of the real HDTB / UP_Hindi treebank in this repo (see
**Limitations**). `data/raw/toy_train.conllu` / `toy_val.conllu` are a small,
hand-built CoNLL-U corpus used to exercise and demonstrate the full training
loop end-to-end:

```bash
python scripts/train.py \
  --train data/raw/toy_train.conllu \
  --val data/raw/toy_val.conllu \
  --epochs 80 \
  --checkpoint-dir checkpoints
```

This writes `checkpoints/final_model.pt` (always, at the end of training) and
`checkpoints/best_model.pt` (only if `--val` is given and at least one epoch
improves validation Macro-F1). To train on your own CoNLL-U corpus instead,
point `--train`/`--val` at it — see
[`src/data_pipeline/hdtb_parser.py`](neuro_symbolic_srl/src/data_pipeline/hdtb_parser.py)
for the expected format (standard CoNLL-U with Paninian Karaka labels — `k1`,
`k2`, `k3`, `k4`, `k5`, `k7p`, `k7t` — in the DEPREL column, plus an optional
`# predicate_synset_id = <int>` comment per sentence for WSD supervision).

Hyperparameters (learning rates, batch size, epochs, early-stopping patience,
λ for the WSD loss term) live in
[`config/training_config.yaml`](neuro_symbolic_srl/config/training_config.yaml);
architecture sizes (BiLSTM hidden dim, dropout, backbone name, label set) live
in [`config/model_config.yaml`](neuro_symbolic_srl/config/model_config.yaml).

## Exporting to ONNX + INT8 quantization

```bash
python scripts/export_onnx.py          # reads checkpoints/final_model.pt, writes checkpoints/srl_net_int8.onnx
```

Takes ~2 minutes; the ~1 GB intermediate FP32 graph is deleted afterwards
unless you pass `--keep-fp32`. `--checkpoint` and `--output-dir` override the
defaults.

Only the encoder + role-scorer MLP are traced to ONNX; the ontology mask
generation and Viterbi search stay in native Python (dynamic lexicon lookups
and graph search don't belong in a static ONNX graph), and the WSD head runs
separately in PyTorch against the exported graph's pooled predicate-hidden
output.

## Running the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Requires `checkpoints/srl_net_int8.onnx` (or set `SRL_ONNX_MODEL_PATH` to
point elsewhere) and, optionally, `checkpoints/final_model.pt` for trained
CRF/WSD weights (`SRL_CHECKPOINT_PATH`) — without a checkpoint the service
still starts and serves requests, just with randomly-initialized CRF/WSD
weights layered under the (still-enforced) ontology mask.

```bash
curl -X POST http://localhost:8000/api/v1/parse_frame \
  -H "Content-Type: application/json" \
  -d '{"text": "पुलिस ने दूरबीन से चोर को देखा"}'
```

```json
{
  "predicate": {"text": "देखा", "lemma": "देखना", "sense": "...", "synset_id": 6229},
  "arguments": [
    {"span": [0, 1], "text": "पुलिस ने", "karaka": "k1", "role": "ARG0"},
    {"span": [2, 3], "text": "दूरबीन से", "karaka": "k3", "role": "ARGM-INS"},
    {"span": [4, 5], "text": "चोर को", "karaka": "k2", "role": "ARG1"}
  ]
}
```

`target_predicate` (0-indexed word position) is optional in the request body;
if omitted, the last token is used as a simple SOV-order heuristic (this
service has no dependency parser to identify the verb properly — see
**Limitations**).

## Benchmarking latency

```bash
python scripts/benchmark_latency.py --onnx-model checkpoints/srl_net_int8.onnx --num-requests 1000
```

Measures p50/p95/p99 CPU latency of the ONNX graph alone (not ontology
masking/decoding, which are comparatively cheap). `--intra-op-threads`
defaults to 1 — in testing, more threads improved mean/median latency but
made p95/p99 *worse* due to threading overhead on such short sequences; try
both and see what your machine does.

## Known limitations

Documented here instead of glossed over, since they materially affect what
you should expect:

- **No real treebank included.** `data/raw/toy_*.conllu` is a small,
  hand-built corpus (a handful of sentences with real, looked-up IndoWordNet
  gold synset IDs), not HDTB/UP_Hindi. It's enough to demonstrate the full
  training loop converges and to overfit the three verification sentences
  from `context.txt` §6 — it is **not** enough to produce the F1 numbers in
  `context.txt` §7. Training on the real corpus is left to you (no confirmed
  public download URL for it was available in the environment this was built in).
- **No lemmatizer or dependency parser.** `api/main.py`'s raw-text endpoint
  whitespace-tokenizes input and uses small, dictionary-*validated*
  suffix-stripping guessers (`src/data_pipeline/lemmatizer.py`) to recover
  verb infinitives and masculine-oblique noun forms — not real morphological
  analysis. It covers the common regular cases (and is validated against the
  real IndoWordNet lexicon before being trusted) but not, e.g., suppletive
  verbs (जाना → गया) or other noun declension classes. The CoNLL-U training/
  eval path doesn't have this problem — it always uses the corpus's own gold
  lemma column.
- **`ai4bharat/indic-bert` is gated** on Hugging Face as of when this was
  built (403 without manual access approval), so the default backbone is
  `google/muril-base-cased` instead (same 768-dim, openly accessible). Swap
  `encoder.backbone_name` in `config/model_config.yaml` if you have indic-bert
  access and prefer it.
- **Latency/size targets are close but not met**, because the backbone is a
  full-size pretrained transformer (178M params) rather than a small
  distilled model: quantized INT8 ONNX is ~258 MB (target: ≤160 MB) and CPU
  p95 latency is ~20.8ms single-threaded (target: ≤20ms). Both numbers are
  real, measured results (see **Benchmarking latency**), not adjusted to hit
  the target.
- **IndoWordNet's free data dump has sparse hypernym coverage** (~9% of
  synsets have a recorded HYPERNYMY relation). `src/ontology/hypernym_engine.py`
  compensates with gloss/lemma keyword matching (plus a regex for the common
  "one who does X" agentive-noun gloss idiom), documented in
  `config/ontology_rules.yaml`.
- **Not yet built**: `src/utils/metrics.py`, `src/utils/logger.py`,
  `scripts/evaluate.py`, `scripts/download_resources.sh`, and a `Dockerfile`
  are named in `context.txt`'s directory layout but weren't needed by
  anything implemented so far; they're empty/absent rather than stubbed out.
