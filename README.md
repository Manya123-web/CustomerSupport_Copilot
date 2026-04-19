# Customer Support Copilot

A document-grounded Retrieval-Augmented Generation (RAG) system that answers customer questions from a knowledge base, opens a support ticket when an account action is needed, and escalates to a human rather than guessing when it is unsure.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manya123-web/CustomerSupport_Copilot/blob/main/notebook/colab.ipynb)

---

## Overview

Given a user query, the system does one of three things:

1. **Answers from the knowledge base.** Retrieves the most relevant passages, generates an answer, and cites the source document.
2. **Opens a support ticket.** When the query is an account-level issue (e.g. "I can't log in"), it skips retrieval and creates a ticket.
3. **Escalates to a human.** When the knowledge base is insufficient or the generated answer is not grounded in the retrieved evidence, the system opens a ticket instead of hallucinating an answer.

The third behaviour — knowing when *not* to answer — is the central design point of the project.

---

## Features

- Hybrid retriever: dense (bi-encoder) + lexical (Jaccard) with a learned 2-parameter fusion layer (α and β trained with manual gradients).
- Cross-encoder reranker for precision on the top-K candidates.
- Generator: Flan-T5-large fine-tuned with DoRA adapters and a DPO-style preference objective that prefers citation-prefixed answers.
- Conversation memory: bounded per-session buffer that resolves pronoun references and injects dialogue history into the prompt.
- Confidence-Gated Retrieval Augmentation (CGRA): a composite confidence score that decides when to fall back to a web scraper and when to escalate.
- Post-generation grounding check: ungrounded answers are converted to tickets.
- Pydantic-validated tool schema: every tool call is type-checked before execution.
- Two routers: a fast keyword matcher (default) and an opt-in learned logistic-regression router.
- Baseline-vs-full evaluation on a fixed test set, with JSON result dumps and a delta table.
- Built-in diagnostics for overfitting, underfitting, and train/test leakage.
- 30 curated demo questions across four behaviour categories.

---

## Tech stack

- **Language**: Python 3.10+
- **Deep learning**: PyTorch 2.1+, Transformers 4.40+, PEFT 0.9+ (DoRA), Sentence-Transformers, Accelerate 1.0+
- **Retrieval**: FAISS (CPU)
- **Serving**: FastAPI, Uvicorn
- **Validation**: Pydantic 2
- **Metrics**: scikit-learn, NLTK, rouge-score
- **Notebook**: Jupyter (for chunking + Colab end-to-end run)

---

## Folder structure

```
project/
├── config/
│   ├── config.yaml         shared hyperparameters (loaded by every run)
│   ├── baseline.yaml       frozen encoder, no reranker, no DoRA, no CGRA
│   └── full.yaml           everything enabled
├── data/
│   ├── raw/
│   │   ├── multidoc2dial_raw.json   ← drop your dataset here
│   │   └── policy_db.json           static policy DB (data, not code)
│   ├── processed/          chunks.json, faiss_*.bin, fusion_weights.json (generated)
│   └── splits/             train.json + test.json (generated, 80/20 doc-level)
├── models/
│   ├── base_model.py       Flan-T5 loader with strict DoRA attach
│   ├── embeddings.py       bi-encoder load + MNRL fine-tune
│   └── quantization.py     fp16 / fp32 / optional int8 helpers
├── retrieval/
│   ├── retriever.py        FAISS + LearnableFusion (manual gradients)
│   └── reranker.py         cross-encoder reranker
├── training/
│   ├── train.py            CLI: --stage biencoder | fusion | dora | router | all
│   ├── evaluation.py       CLI: --config ... | --compare
│   ├── agent.py            CGRA gate + router + linear agent loop
│   ├── tools.py            pydantic-validated SearchKB / GetPolicy / CreateTicket
│   ├── router.py           learned logistic-regression router (opt-in)
│   ├── memory.py           per-session conversation memory
│   ├── diagnostics.py      overfit / underfit / leakage checks
│   └── demo_questions.py   30 curated demo questions, 4 categories
├── utils/
│   ├── metrics.py          P@K, R@K, MRR, BLEU, ROUGE-L, grounding, CER
│   ├── schema.py           pydantic ToolCall + AgentResponse
│   ├── config.py           YAML loader with `extends:` inheritance
│   └── logging.py          JSON-line run logger
├── demo/
│   ├── app.py              FastAPI POST /query
│   └── cli.py              interactive terminal demo
├── notebook/
│   ├── main.ipynb          local thin orchestrator (chunking + eval)
│   └── colab.ipynb         end-to-end Google Colab notebook
├── tests/                  pytest suite (offline, runs in seconds)
├── experiments/results/    JSON metric dumps from each run
├── requirements.txt
├── Makefile                top-level entry points
└── README.md
```

---

## Requirements

- Python 3.10 or newer
- ~6 GB free disk space for the model checkpoints
- **GPU optional but strongly recommended for training.** Inference also runs on CPU but is much slower.
  - Colab T4 (free tier) is sufficient for both training and evaluation.
  - Locally: any NVIDIA GPU with ≥6 GB VRAM works in fp16.
- Internet access on first run (to download base models from Hugging Face).

All Python dependencies are pinned in `requirements.txt`.

---

## Setup

### Option A — Google Colab (recommended for first-time users)

1. Click the **Open in Colab** badge at the top of this README. The notebook is `notebook/colab.ipynb`.
2. `Runtime` → `Change runtime type` → select **T4 GPU**.
3. Run the cells top-to-bottom. The notebook will:
   - install dependencies,
   - clone this repo into `/content/project` (or use a zip if you have one),
   - download the MultiDoc2Dial dataset from Hugging Face,
   - build chunks and the 80/20 train/test split,
   - (optionally) train the bi-encoder and DoRA adapters,
   - run baseline and full evaluation,
   - print the comparison table,
   - run diagnostics and the 30 curated demo questions.

### Option B — Local machine

```bash
git clone https://github.com/Manya123-web/CustomerSupport_Copilot.git
cd CustomerSupport_Copilot

# 1) Install pinned dependencies + NLTK data
make install

# 2) Provide the dataset
#    Drop multidoc2dial_raw.json into data/raw/
#    (the Colab notebook can also fetch it from Hugging Face)

# 3) Build chunks and the 80/20 train/test split
#    This runs notebook/main.ipynb headless; it writes:
#      data/processed/chunks.json
#      data/splits/train.json
#      data/splits/test.json
make chunks
```

After `make chunks` you should see populated files under `data/processed/` and `data/splits/`.

---

## How to run

The pipeline has four stages: **install → chunk → train → evaluate**, plus optional demo / API serving.

### 1. Train

Each stage can be run independently or all together.

```bash
make train-biencoder   # ~5 min on T4   — fine-tunes the dense retriever
make train-fusion      # <1 min          — fits α/β for hybrid fusion
make train-dora        # ~15–20 min on T4 — DoRA adapters + DPO preference loss
make train-router      # <1 min          — opt-in learned router
make train             # all of the above, in order
```

Checkpoints are written to `models/checkpoints/`.

### 2. Evaluate

Run the baseline first, then the full system, then the comparison.

```bash
make eval-baseline   # baseline config (no reranker, no DoRA, no CGRA)
make eval-full       # full config (everything enabled)
make compare         # prints baseline-vs-full delta table
make eval            # all three of the above, in order
```

Each run writes a timestamped JSON file under `experiments/results/`. `make compare` reads the latest baseline and full runs and writes `experiments/results/comparison_latest.json`.

> **Note:** evaluation is read-only by design. If a required artifact (fusion weights, learned router) is missing, the run will stop with an actionable error pointing at the right `make train-*` step. To override for a one-off run, see [Troubleshooting](#troubleshooting).

### 3. Run the offline test suite

```bash
make test
```

The pytest suite covers metrics, the agent router, schema validation, the policy DB, the policy mapper, conversation memory, and the demo question manifest. It runs in a few seconds and does not require a GPU.

### 4. Try it interactively

```bash
make demo   # interactive terminal — type a query, see the answer + tool trace
make api    # FastAPI server at http://localhost:8000
```

API example:

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "What documents do I need to apply for retirement benefits?",
       "session_id": "user-42"}'
```

The same `session_id` on a follow-up query enables pronoun resolution and dialogue context.

### 5. Run diagnostics

```python
from training.diagnostics import run_diagnostics
run_diagnostics("config/full.yaml")
```

Prints a single report covering: train/test document and chunk disjointness, retrieval P@5/MRR on both splits, fusion weight sanity (α, β), learned router train-vs-paraphrase generalisation, and a literal-WEB_CACHE leakage check.

---

## Output and results

`experiments/results/<config_name>_<timestamp>.json` contains the full metric dump for each run. The `compare` step also writes `comparison_latest.json` with the side-by-side delta.

Metrics computed on the held-out test set:

| Group | Metric | Meaning |
|---|---|---|
| Retrieval | P@5, P@10, P@15, P@20 | Precision at K |
| Retrieval | R@5, R@10, R@15, R@20 | Recall at K |
| Retrieval | MRR | Mean reciprocal rank of the first relevant document |
| Generation | BLEU | n-gram overlap with the reference |
| Generation | ROUGE-L | longest-common-subsequence F-measure |
| Faithfulness | grounding | fraction of non-stop answer tokens that appear in retrieved context |
| Workflow | escalation_rate | fraction of queries that opened a ticket |
| Workflow | CER | Correct Escalation Rate (project-specific) — on a KB-sufficient set this equals `1 − escalation_rate` |
| Latency | latency_p50_ms, latency_p95_ms | per-query inference latency percentiles |

Reference results on the 20 hand-written MultiDoc2Dial queries:

| Metric | Baseline | Full | Δ |
|---|---|---|---|
| P@5 | 0.32 | 0.46 | +0.14 |
| Grounding | 0.52 | 0.66 | +0.14 |
| CER | 0.65 | 0.95 | +0.30 |
| Escalation rate | 0.35 | 0.05 | −0.30 |

---

## What each script does and when to run it

| Script | When | Purpose |
|---|---|---|
| `notebook/main.ipynb` (via `make chunks`) | once, after first install | Reads raw JSON, builds 150-word chunks, writes the 80/20 doc-level split |
| `training/train.py --stage biencoder` | once per dataset change | Fine-tunes MiniLM with MultipleNegativesRankingLoss |
| `training/train.py --stage fusion` | after biencoder | Fits the 2-parameter α/β hybrid fusion weights via BPR |
| `training/train.py --stage dora` | once per dataset change | DoRA adapters on Flan-T5-large with DPO preference loss |
| `training/train.py --stage router` | optional | Trains the learned logistic-regression router |
| `training/evaluation.py --config ...` | after training | Builds the system from a config and writes a JSON metric dump |
| `training/diagnostics.py` | any time after training | Overfitting / underfitting / leakage report |
| `demo/cli.py` | any time after training | Interactive terminal demo |
| `demo/app.py` | any time after training | FastAPI server with per-session memory |

---

## Troubleshooting

### Colab fails to find the project directory

The `Open in Colab` badge serves the notebook from GitHub with no project files on the VM. Cell 6 of `notebook/colab.ipynb` now `git clone`s the repo automatically as a fallback. If the clone fails (network restrictions, fork URL drift), upload `CustomerSupportCopilot_restructured.zip` via Colab's file pane and re-run the cell.

### Eval stops with `Fusion weights not found`

Evaluation is read-only. Run training first:

```bash
make train-fusion
```

Or for a one-off override:

```bash
COPILOT_ALLOW_EVAL_TRAIN=1 make eval-full
```

### Eval stops with `Learned router checkpoint not found`

Same policy as above. Either run `make train-router`, or set `agent.router_autotrain_in_eval: true` in the config, or export `COPILOT_ALLOW_EVAL_TRAIN=1` for a one-off run.

### GPU shows zero load while encoding takes minutes

The bi-encoder and cross-encoder are now constructed with explicit `device="cuda"` placement, so this should no longer happen. If you still see CPU-only behaviour, confirm `torch.cuda.is_available()` returns `True` and that PyTorch was installed with CUDA support (not the CPU-only wheel).

### `RuntimeError: Expected all tensors on the same device`

This was caused by `device_map="auto"` partially placing the Flan-T5 model under some accelerate versions. The current `models/base_model.py` does an explicit `model.to("cuda")` after load and prints the actual device of the first parameter at startup. If you still see this, check the `[base_model]` log line for the device string.

### NLTK `punkt_tab` download fails

`punkt_tab` requires NLTK ≥ 3.9. If you see a `LookupError`, upgrade NLTK:

```bash
pip install --upgrade "nltk>=3.9"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

The agent has a regex sentence-split fallback, so requests will still complete, but quality is mildly degraded without `punkt_tab`.

### DoRA training loss is NaN on Colab T4

Flan-T5-large is known to overflow in pure fp16. The DPO loop runs the model forward in fp16 but computes the loss in fp32 (no autocast). If you still see NaNs, switch the generator dtype to `float32` in `config/config.yaml` (`generator.dtype: "float32"`) — slower but stable.

---

## Notes on what this system does not do

- It does **not** continuously learn from user interactions. Training is offline; serving is read-only inference.
- The grounding score is **lexical token overlap**, not entailment. It catches obviously ungrounded answers but does not catch correctly-paraphrased-but-wrong answers. See `agent.py::_grounding` for the call site if you want to swap in an NLI model.
- The 30-question demo set and the 20-query evaluation set are small by design — they are illustrative, not a benchmark.

---

## Future improvements

- Replace the lexical grounding check with an NLI-based entailment classifier.
- Add a real validation split (currently the project ships only train/test) and use it for early-stopping the bi-encoder and threshold tuning for grounding.
- Cache encoded chunk embeddings on disk so eval does not re-encode every run.
- Add classification metrics (accuracy / precision / recall / F1) on the routing and escalation decisions to complement the retrieval-side metrics.
- Add an end-to-end smoke test target (`make smoke`) that runs one query through `build_system → run_agent` against a tiny fixture so CI can fail fast on integration regressions.

---

## License

See repository for licensing information.
