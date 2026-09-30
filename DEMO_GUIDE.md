# NeuroKaraka — Demo & Presentation Guide

Everything you need to (1) get this running on a different machine (written
for macOS, with Windows/Linux notes), (2) run the live demo, and (3) present
it: a timed script plus prepared answers to likely questions.

> **Honesty note on Mac:** this project was built and tested on Windows 11.
> The code has no OS-specific paths (it uses `pathlib` throughout, and the
> one Windows-only encoding workaround is harmless elsewhere), but I have
> **not** run it on macOS. Do the full setup + verification on the Mac
> **the day before**, not the morning of.

---

## 0. The one thing to know first

A `git clone` alone will **not** give you a working demo, for two reasons:

1. **Some files aren't committed yet.** As of this writing, the API
   (`api/main.py`, `api/schemas.py`), the lemmatizer, the ONNX exporter,
   `scripts/demo.py`, `scripts/export_onnx.py`, `scripts/benchmark_latency.py`
   and three test files are **untracked**, and `train.py`/`pipeline.py`/
   `test_morphology.py` have uncommitted edits. Commit and push them from the
   Windows machine first (Section 1).
2. **Trained weights are git-ignored** (`*.pt`, `*.onnx` in `.gitignore`;
   `final_model.pt` is ~1 GB). You must either copy the weights across or
   retrain on the Mac (Section 3).

---

## 1. On the Windows machine (before you leave it)

```bash
cd NeuroKaraka
git status                       # see what's untracked/modified
git add DEMO_GUIDE.md README.md neuro_symbolic_srl
git status                       # confirm NO .venv/, *.pt, *.onnx in the staged list
git commit -m "Add API, ONNX export, lemmatizer, demo script, tests, guides"
git push
```

Then copy these to a USB stick / cloud drive / AirDrop-able location:

| File | Size | Needed for |
|---|---|---|
| `neuro_symbolic_srl/checkpoints/final_model.pt` | ~1 GB | the demo **and** re-exporting the API model |
| `neuro_symbolic_srl/checkpoints/srl_net_int8.onnx` | ~260 MB | the API (optional — can be regenerated from `final_model.pt` in ~2 min) |

Skip `srl_net.onnx` (1 GB, intermediate FP32 graph) — not needed.

---

## 2. Setup on the Mac, from scratch

### 2.1 Check your machine

```bash
uname -m          # arm64 = Apple Silicon (good).  x86_64 = Intel Mac (see warning)
python3 --version # need 3.10 or newer
```

> **Intel Mac warning:** recent PyTorch releases no longer publish macOS
> x86_64 wheels, and this project was built against a current PyTorch. On an
> Intel Mac `pip install` may fail to find a matching `torch`. I haven't
> tested a workaround (pinning an older `torch`/`numpy<2`). If you have a
> choice of machine, use Apple Silicon or a Linux/Windows laptop.

If `python3` is older than 3.10 (macOS often ships 3.9):

```bash
# install Homebrew first if needed: https://brew.sh
brew install python@3.12
# then use `python3.12` in place of `python3` below
```

### 2.2 Get the code

```bash
git clone https://github.com/ritwikmohanty/NeuroKaraka.git
cd NeuroKaraka/neuro_symbolic_srl       # <- every command below runs from here
```

(If the repo is private, git will ask you to authenticate.)

### 2.3 Create the environment and install

```bash
python3 -m venv .venv
source .venv/bin/activate               # prompt should now show (.venv)
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs everything (torch, transformers, onnxruntime, fastapi,
sentence-transformers, pyiwn, pytest, …) from `pyproject.toml`. Expect a few
minutes and ~1–2 GB of packages.

### 2.4 First-run downloads (~3 GB, do on good wifi)

Nothing to do manually — they trigger automatically the first time the code
runs, then are cached:

| Resource | Size on disk | Cached at |
|---|---|---|
| IndoWordNet Hindi data | ~140 MB | `~/iwn_data` |
| `google/muril-base-cased` | ~1.8 GB | `~/.cache/huggingface/hub` |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | ~1.1 GB | `~/.cache/huggingface/hub` |

Trigger all three now with the quick test in Section 4 so nothing downloads
during the presentation.

---

## 3. Getting the trained weights onto the Mac

### Option A — copy the weights (fastest, recommended)

Put `final_model.pt` into `neuro_symbolic_srl/checkpoints/` (create the folder
if missing: `mkdir -p checkpoints`). Then generate the API model locally:

```bash
python scripts/export_onnx.py
# -> prints sizes, writes checkpoints/srl_net_int8.onnx (~258 MB), deletes the
#    1 GB intermediate FP32 graph. Takes ~2 minutes.
```

(Copying `srl_net_int8.onnx` across instead also works in principle, but
re-exporting on the Mac removes any doubt about cross-platform ONNX
compatibility.)

### Option B — retrain on the Mac (no file copying)

```bash
mkdir -p checkpoints
python scripts/train.py \
  --train data/raw/toy_train.conllu \
  --epochs 80 \
  --checkpoint-dir checkpoints
python scripts/export_onnx.py
```

Training took roughly 10–20 minutes on my Windows laptop CPU; I haven't
measured it on Apple Silicon. Early epochs show loss around 19–20; by epoch
80 it should be well under 1. Training reuses the tiny hand-built corpus in
`data/raw/` (tracked in git), so nothing else is needed. Results will be
very close to, but not bit-identical with, the Windows-trained weights —
**re-run the demo and confirm all three sentences still come out right**
before trusting it on stage.

---

## 4. Verify everything works (do this the day before)

```bash
source .venv/bin/activate        # if not already active
export HF_HUB_OFFLINE=1          # only AFTER the first run has cached everything
pytest tests/ -q                 # expect: 130 passed
python scripts/demo.py
```

- The **first** run (without `HF_HUB_OFFLINE`) downloads the ~3 GB of models
  and is slow; subsequent runs are fast. Run `pytest` once with no offline
  flag, then use `HF_HUB_OFFLINE=1` afterwards.
- If `test_pipeline.py` / `test_api.py` show as *skipped*, the checkpoint or
  ONNX file is missing — go back to Section 3.
- `demo.py` should print the three sentences exactly as shown in Section 5.

---

## 5. Running the demo

### 5.1 The core demo (this is the money shot)

```bash
python scripts/demo.py
```

Expected output (trimmed):

```
[1] चाबी ने ताला खोला  ("The key opened the lock")
  >> Neural network ALONE (no ontology mask):
    चाबी       B-ARG0          <-- the trap: ergative ने => "agent"
    ने         I-ARGM-INS
    ताला       B-ARG1
  >> FULL system (+ IndoWordNet ontology mask):
    चाबी       B-ARGM-INS      <-- corrected: IndoWordNet says चाबी is inanimate
    ने         I-ARGM-INS
    ताला       B-ARG1
  *** The mask changed the outcome ... ***

[2] सेब को बच्चे ने खाया   -> सेब को = ARG1,  बच्चे ने = ARG0     (scrambled OSV)
[3] पुलिस ने दूरबीन से चोर को देखा
                          -> पुलिस ने = ARG0, दूरबीन से = ARGM-INS, चोर को = ARG1
```

The raw-network-vs-full-system comparison on sentence 1 is your strongest
evidence: it shows the neural model failing by itself and the symbolic layer
fixing it, live.

### 5.2 (Optional) Run it as a real service

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Wait ~10–15 s for models to load (you'll see "Application startup complete").
Then open **http://127.0.0.1:8000/docs** in a browser → `POST
/api/v1/parse_frame` → **Try it out** → body:

```json
{ "text": "चाबी ने ताला खोला" }
```

Expected response: `खोला` → lemma `खोलना`, synset 10852; `चाबी ने` = `ARGM-INS`
(Karaka `k3`), `ताला` = `ARG1`.

Or from a terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/parse_frame \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"text": "चाबी ने ताला खोला"}'
```

> On **Windows Git Bash** I saw Devanagari in inline `curl -d '...'`
> arguments get corrupted to `????` (a shell-encoding issue, not a server
> bug). macOS Terminal is UTF-8 by default so it should work, but the
> browser `/docs` form is the safest option for a live demo either way.

Stop the server with `Ctrl+C`.

### 5.3 (Optional) Proof of rigor

```bash
pytest tests/test_pipeline.py tests/test_api.py -v
```

### 5.4 (Optional) Latency benchmark

```bash
python scripts/benchmark_latency.py --onnx-model checkpoints/srl_net_int8.onnx
```

My Windows numbers: p50 ≈ 18.1 ms, **p95 ≈ 20.8 ms** (target ≤ 20 ms — narrowly
missed), p99 ≈ 21.6 ms. **Your Mac will give different numbers** — run it
beforehand and quote *your* measured figures, not mine.

---

## 6. Presentation script (≈ 8 minutes + Q&A)

*Written for you, presenting to a grader. Read it aloud at least twice;
adapt the wording to sound like you.*

| Segment | Time |
|---|---|
| Problem | 0:30 |
| Architecture | 1:30 |
| Live demo | 2:30 |
| Engineering depth | 2:00 |
| Limitations | 0:45 |
| Close | 0:15 |

### Problem (30 s)

> "Standard neural sequence taggers fail in a predictable way on Hindi. If a
> model sees the postposition *ने*, it learns 'ने means agent' — because
> that's true most of the time in training data. But it isn't always true.
> In 'चाबी ने ताला खोला' — 'the key opened the lock' — चाबी carries *ने*, but
> a key can't intentionally open anything. It's the instrument, not the
> agent. A pure neural model gets this wrong. I built a system that doesn't."

### Architecture (90 s)

> "The architecture has two halves that don't blindly trust each other. A
> BiLSTM over a MuRIL transformer reads the sentence and scores every
> possible role at every token — that's the neural half, and it's exactly
> the part that falls for the ने-means-agent trap. Separately, an
> IndoWordNet ontology lookup asks: is this word's referent even *capable*
> of being an agent? Is चाबी animate? No. So a symbolic mask forces the
> score for 'चाबी = agent' down by ten thousand — a hard constraint, not a
> soft nudge — before a Viterbi decoder picks the best whole-sentence
> labelling under that constraint. The neural half proposes; the symbolic
> half disposes."

### Live demo (2½ min)

*Run `python scripts/demo.py`.*

> "Let me show you this isn't hypothetical. Watch sentence 1. First I print
> what the neural network predicts completely on its own, no mask." *(pause
> on the output)* "It says चाबी is `B-ARG0` — the agent. That's the exact
> failure I just described, happening live on my own trained model. Now the
> same sentence through the full system." *(point)* "`ARGM-INS` —
> instrument. Correct. The ontology mask changed the decoded output, live,
> in front of you."
>
> "Sentence 2 tests word-order scrambling: 'सेब को बच्चे ने खाया' — 'the
> apple, the child ate'. The apple comes first even though it's the thing
> being eaten. A model that just learned 'first noun = subject' gets this
> backwards. Mine doesn't, because it's using the case markers and
> IndoWordNet's animacy check on both nouns — the mask forbids the apple
> from being the eater."
>
> "Sentence 3 is an attachment ambiguity: 'the police saw the thief with a
> telescope.' Does the telescope belong to the seeing, or to the thief? The
> system attaches it to the verb, as the instrument of seeing."

*(Optional, if time: show the `/docs` page and POST sentence 1.)*

### Engineering depth (2 min) — this is what separates a working demo from a grade

> "Three things that aren't visible from the demo.
>
> First: the spec I was working from named a specific Hindi transformer,
> indic-bert. I checked — Hugging Face now gates it behind manual approval.
> I verified that empirically and switched to MuRIL: same 768 dimensions,
> openly accessible.
>
> Second: IndoWordNet's free data only records explicit hypernym — 'is-a' —
> chains for about 9% of entries; I measured 3,577 out of 40,360. So I
> couldn't just walk up an 'is-a' tree to check animacy. I combined that
> with dictionary-gloss keyword matching, and I found a specific gap:
> professions like 'police' and 'thief' aren't defined with an animacy word
> at all — they're defined as 'one who does X'. I wrote a pattern for that
> idiom and validated it against the real dictionary.
>
> Third — the one I'm proudest of catching. When I wired this into a
> raw-text API with no lemmatizer, the exact same चाबी sentence started
> failing again, silently: the inflected verb 'खोला' doesn't match the
> dictionary entry 'खोलना' the rule is keyed on. I built a small
> dictionary-validated lemma guesser to close that gap, and confirmed it
> against real IndoWordNet lookups rather than assuming it worked."

### Honest limitations (45 s) — don't skip this

> "Two honest caveats. One: I trained on a small hand-built corpus of five
> sentences, not the full Hindi treebank, because I didn't have a verified
> way to download it. So this demo proves the training loop and the whole
> pipeline are correct end to end — it is not a claim about accuracy on
> unseen Hindi. Two: the spec's latency target was under 20 milliseconds;
> I measured [your number] p95 with real timing code on the full-size
> backbone. Close, but not there — a distilled model would close the gap."

### Close (15 s)

> "The core idea: the neural network is fast and flexible but gets fooled by
> surface patterns; the symbolic layer is laborious to build but never gets
> fooled by them. Putting a hard constraint layer between the neural scorer
> and the final decision gets you both."

---

## 7. Likely questions, with prepared answers

**Why not just prompt an LLM?**
> An LLM is also a soft statistical estimator — it can be prompted into
> getting this right sometimes, but nothing *guarantees* it respects
> animacy the way a hard −10,000 mask does. This runs in tens of
> milliseconds on CPU, and its constraint is checkable and deterministic.

**What's your accuracy / F1?**
> "I don't have a trustworthy F1 number — that needs the full treebank, which
> I couldn't verify access to. What I can show is a fully implemented
> architecture, a constraint mechanism demonstrated on three linguistically
> motivated cases, and 130 unit and integration tests."

**You trained on these exact sentences — isn't the demo just memorization?**
> "Yes, for the core-role labels on these sentences — I said so up front; it
> proves the pipeline works, not generalization. But the interesting part
> isn't memorized: the no-mask run gets sentence 1 *wrong* even after
> training on it, and the final correct answer comes from a live IndoWordNet
> lookup at inference time."

**Wait — if you trained on the correct label, why does the raw network still say ARG0?**
> "Because training ran with the mask *in the loop*. The loss is computed on
> the masked scores, and a −10,000 entry contributes essentially zero
> gradient — so the network never had to learn to suppress ARG0 for चाबी;
> the symbolic constraint handled it during training and handles it at
> inference. That's by design: the knowledge is a guarantee from the
> ontology, not something the network has to rediscover."

**Why is the raw output `B-ARG0` followed by `I-ARGM-INS` — that's not valid BIO.**
> "Right — the CRF's transition scores are soft preferences, so unconstrained
> decoding can produce ill-formed tag sequences. That's actually more
> evidence for the approach: hard constraints give guarantees that soft
> learned scores don't."

**Why BiLSTM on top of a transformer?**
> "It's the architecture the spec calls for: a recurrent layer gives a strong
> inductive bias for local, strictly ordered structure — like a case marker
> immediately following its noun — on top of the transformer's contextual
> embeddings."

**How does it resolve the telescope attachment ambiguity?**
> "Through the role scorer plus the ontology: IndoWordNet classes दूरबीन as an
> instrument, and the decoder assigns it the instrument role relative to the
> predicate rather than attaching it to the noun." *(Be honest if pressed:
> I verified this on the test sentence, not on a broad attachment benchmark.)*

**What would you do with more time?**
> "Train on the real treebank; replace the heuristic lemma guesser and the
> 'last token is the verb' default with a proper lemmatizer and POS tagger;
> use a smaller distilled backbone to hit the latency target cleanly."

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Not in `neuro_symbolic_srl/`, venv not active, or `pip install -e ".[dev]"` not run |
| `FileNotFoundError: checkpoints/final_model.pt` | Weights missing — Section 3 |
| API crashes at startup: ONNX file not found | Run `python scripts/export_onnx.py` |
| `test_pipeline` / `test_api` are skipped | Same — checkpoint / ONNX missing |
| `pip` can't find `torch` | Python < 3.10, or Intel Mac (Section 2.1) |
| Everything is very slow the first time | ~3 GB of first-run downloads; subsequent runs are fast |
| Hangs/fails with no network after first run | `export HF_HUB_OFFLINE=1` (only once everything is cached) |
| `Address already in use` (port 8000) | `lsof -i :8000` then kill that PID, or use `--port 8001` |
| Devanagari shows as `????` in curl (Windows Git Bash) | Shell encoding — use the `/docs` browser form, or `-d @file.json` |
| Demo labels differ slightly after retraining on the Mac | Weights aren't bit-identical to the Windows run — check all three sentences and re-tune epochs, or copy the original `final_model.pt` (Option A) |

---

## 9. Day-of checklist

- [ ] Ran Section 4 on the presentation laptop itself, with **no network dependence**
- [ ] `HF_HUB_OFFLINE=1` set in the terminal you'll present from
- [ ] Terminal font large enough to read Devanagari from the back of the room
- [ ] `python scripts/demo.py` run once right before presenting (warm caches)
- [ ] Screenshot of the demo output saved as a fallback slide
- [ ] Latency figure in the script replaced with **your** measured number
- [ ] If showing the API: server already started and `/docs` tab already open
- [ ] Laptop charger, and a copy of `final_model.pt` on a USB stick
