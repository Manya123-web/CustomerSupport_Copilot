# Customer Support Copilot

A document-grounded RAG system that answers customer questions from a knowledge base, calls the right support tool when needed, and escalates to a human rather than guessing when it is not sure.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manya123-web/CustomerSupport_Copilot/blob/main/notebook/colab.ipynb)

---

## What this project does

Imagine a customer-support chatbot for a large agency (we use Social Security documents). A user asks a question; three things can happen:

1. **The answer is in the knowledge base.** The bot retrieves the right passages, writes an answer, and shows which document it came from.
2. **The question is an account issue** (login broken, password reset, etc.). The bot skips search and opens a support ticket.
3. **The knowledge base does not actually have the answer.** Most chatbots guess. Ours opens a ticket instead — it would rather escalate than hallucinate.

That third behaviour is the point of the project.

---

## Key features

- Hybrid retrieval with a 2-parameter fusion layer (semantic + keyword similarity) whose weights are learned from data rather than guessed
- Cross-encoder reranking for precision
- Flan-T5-large generator fine-tuned with DoRA + DPO to prefer citation-prefixed answers
- **Conversation memory** — bounded session buffer resolves follow-up references and injects dialogue history into the generation prompt
- Confidence-Gated Retrieval Augmentation (CGRA) — composite confidence score that decides when to call the web scraper and when to escalate
- Post-generation grounding check that converts ungrounded answers into tickets
- Pydantic-validated tool schema — every tool call is type-checked before execution
- Two routers: fast keyword matching (default) or a learned logistic-regression classifier (opt-in, catches paraphrases)
- Baseline-vs-full evaluation on the same test set with JSON result dumps
- Built-in diagnostics for overfitting, underfitting and leakage
- 30 curated test questions across six behaviour categories (simple, complex, multi-turn, edge-case, policy, memory)

---

## Results

On a MultiDoc2Dial held-out set (20 hand-written queries):

| Metric              | Baseline | Full system | Delta    |
| ------------------- | -------- | ----------- | -------- |
| P@5                 | 0.32     | 0.46        | +0.14    |
| Grounding           | 0.52     | 0.66        | +0.14    |
| CER                 | 0.65     | 0.95        | +0.30    |
| Escalation rate     | 0.35     | 0.05        | -0.30    |

---

## Quick start

### Option A — Google Colab (recommended)

Click the "Open in Colab" badge at the top of this README. It opens `notebook/colab.ipynb` directly from this repo. Then:

1. `Runtime` → `Change runtime type` → select **T4 GPU** (free tier)
2. Upload `multidoc2dial_raw.json` to the Colab Files pane, or have it on your Google Drive
3. Run cells top to bottom. The notebook installs dependencies, builds chunks, optionally trains, runs evaluation, runs the diagnostics, and executes the 30-question demo

If the badge does not work, go to [colab.research.google.com](https://colab.research.google.com), open the GitHub tab, paste this repo's URL, and select `notebook/colab.ipynb` from the list.

### Option B — Local machine

```bash
git clone https://github.com/Manya123-web/CustomerSupport_Copilot.git
cd CustomerSupport_Copilot

make install        # installs pinned deps + NLTK data
# Place multidoc2dial_raw.json in data/raw/
# Then run the chunking cell in notebook/main.ipynb

make train          # trains bi-encoder + DoRA adapters (~25 min on T4)
make eval           # runs baseline + full configs, prints delta table
make test           # pytest suite (15 tests, <10s, no GPU required)
make demo           # interactive terminal demo
make api            # FastAPI at http://localhost:8000
```

---

## Overfitting, underfitting and leakage checks

The project ships `training/diagnostics.py` which runs five automatic checks:

- **Train/test document disjointness** — the split is at document level, not chunk level, so no phrase from a training doc can bleed into the test set
- **Retriever train-vs-test probe** — P@5 and MRR on both splits. If train is much higher than test the retriever is memorising; if both are low it isn't learning
- **Fusion-weight sanity** — α and β should not collapse to 1.0/0.0 (one signal dominating) or stick at 0.5/0.5 (no differentiation)
- **DoRA early stopping** — 20% of preference pairs held out as a validation set; training stops when val-loss stops improving for two epochs
- **Cache-leakage scan** — confirms no eval query is a literal WEB_CACHE key

Run it in Colab or locally:

```python
from training.diagnostics import run_diagnostics
run_diagnostics('config/full.yaml')
```

---

## 30 curated demo questions

`training/demo_questions.py` contains 30 hand-written queries split into four categories so you can probe different parts of the system:

| Category         | Count | Expected behaviour                                 |
| ---------------- | ----- | -------------------------------------------------- |
| `STANDARD`       | 8     | SearchKB, grounded answer with citation            |
| `PARAPHRASED`    | 8     | Same intents, colloquial wording — tests retrieval |
| `ACCOUNT_ISSUE`  | 7     | Routes to CreateTicket, no search                  |
| `OUT_OF_SCOPE`   | 7     | CGRA should escalate or augment via web scraper    |

Run all 30 against the chatbot:

```python
from training.demo_questions import all_demo_questions
for item in all_demo_questions():
    r = ask(item['query'])   # ask() is defined in the Colab demo cell
```

---

## Learning model: offline-trained, inference-only at runtime

This system does **not** continuously learn from user interactions. It is trained once using the four fine-tuning stages below, then serves read-only inference afterwards. Specifically:

- No online weight updates during serving
- No feedback collection or incremental adaptation
- User conversations are not stored or used to update the model
- Re-training requires re-running the `make train` pipeline on new labelled data

This is a deliberate design choice for reproducibility and safety. A support copilot that updates itself from live traffic would be hard to audit, prone to feedback loops, and could absorb incorrect user-supplied "corrections". For a production deployment that needs adaptation, the correct approach is to log interactions, review them, and periodically retrain on curated examples — not to close the loop automatically.

---

## Grounding check: what it is and what it isn't

The grounding score used by the post-generation safety net is based on **lexical token overlap** between the answer and the retrieved context. A fraction of the answer's non-stopword tokens that also appear in the context.

What this catches reliably:
- Obviously ungrounded answers (e.g. the model making up a statistic)
- Answers about topics the retriever did not surface
- Free-associating completions far from the evidence

What this does **not** catch:
- Answers that copy many words from the context but misinterpret them (high lexical overlap, wrong meaning)
- Answers that are correctly paraphrased but use different wording (low lexical overlap, correct meaning — penalised)
- Logical errors, arithmetic errors, or wrong inferences from correct evidence

A stronger check would use Natural Language Inference (NLI): feed (answer, context) into an entailment classifier and accept only when the context entails the answer. We keep that upgrade out of v1 because it adds a second ~100 MB model and ~300 ms per query, but it is a drop-in replacement — see `agent.py::_grounding` for the call site. The current implementation is a useful heuristic but should not be treated as a guarantee of factual correctness.

---

## Is evaluation expensive?

No. Weights are not updated during evaluation. Every model is loaded from disk, `.eval()` is called, and inference runs under `torch.no_grad()`.

| Component                       | Trained?                             | Loaded from disk   | Inference cost    |
| ------------------------------- | ------------------------------------ | ------------------ | ----------------- |
| Bi-encoder                      | Fine-tuned once via `make train`     | Yes                | encode() only     |
| Cross-encoder                   | Pre-trained, never modified          | Yes                | predict() only    |
| Flan-T5 + DoRA adapters         | Fine-tuned once via `make train`     | Yes                | generate() only   |
| Learned fusion weights (α, β)   | Trained once, cached to disk         | Yes (after 1st run)| scalar lookup     |
| Learned router (optional)       | Trained via `default_router()`       | Yes (from .npz)    | one matmul        |

The first `make eval` trains the fusion weights and writes them to `data/processed/fusion_weights_ft.json`. Every subsequent evaluation reads that cache in microseconds.

---

## Memory footprint

The codebase uses the following memory-saving techniques:

- Flan-T5 loaded in float16 — halves VRAM (3 GB to 1.5 GB) with no measurable quality drop
- `low_cpu_mem_usage=True` on model load — streams weights instead of duplicating in RAM
- DoRA — only ~0.4% of the LLM's parameters are trainable, the rest stay frozen
- Gradient checkpointing during DoRA training — trades compute for significant VRAM savings
- Fusion query embeddings pre-computed once and cached across all 80 training epochs
- Lexical overlaps pre-cached per triplet
- FAISS `IndexFlatIP` runs on CPU — no GPU VRAM used for retrieval
- Optional INT8 loader (`models/quantized_model.py`) via bitsandbytes
- `torch.cuda.empty_cache()` called between stages
- `@torch.no_grad()` on inference paths so no autograd graph is built

VRAM at inference time on a T4: approximately 1.8 GB. Everything runs on the free Colab tier.

---

## Project layout

```
project/
├── config/
│   ├── config.yaml            shared hyperparameters
│   ├── baseline.yaml          frozen encoder, no reranker, no DoRA, no CGRA
│   └── full.yaml              everything enabled
├── data/
│   ├── raw/
│   │   ├── multidoc2dial_raw.json   (drop your dataset here)
│   │   └── policy_db.json           static policy DB (data, not code)
│   ├── processed/             chunks.json, faiss_*.bin, fusion_weights.json
│   └── splits/                train.json + test.json (80/20 doc-level)
├── models/
│   ├── base_model.py          Flan-T5 loader with strict DoRA attach
│   ├── embeddings.py          bi-encoder load + MNRL fine-tune (Component A)
│   └── quantized_model.py     fp16 + optional INT8
├── retrieval/
│   ├── retriever.py           FAISS + LearnableFusion (manual gradients)
│   └── reranker.py            cross-encoder reranker (Component B)
├── training/
│   ├── tools.py               pydantic-validated SearchKB / GetPolicy / CreateTicket
│   ├── agent.py               CGRA gate + router + linear agent loop
│   ├── router.py              learned logistic-regression router (opt-in)
│   ├── diagnostics.py         overfit / underfit / leakage checks
│   ├── demo_questions.py      30 curated test questions, 4 categories
│   ├── train.py               CLI: --stage biencoder | dora | all
│   └── evaluation.py          CLI: --config ... | --compare
├── utils/
│   ├── metrics.py             P@K, R@K, MRR, BLEU, ROUGE-L, grounding, CER
│   ├── schema.py              pydantic ToolCall + AgentResponse
│   ├── config.py              YAML loader with `extends:`
│   └── logging.py             JSON-line run logger
├── experiments/results/       JSON metric dumps
├── notebook/
│   ├── main.ipynb             thin local orchestrator (imports, no logic)
│   └── colab.ipynb            Google Colab end-to-end notebook
├── demo/
│   ├── app.py                 FastAPI POST /query
│   └── cli.py                 interactive terminal demo
├── tests/                     pytest suite (offline, <10s)
├── report/
│   ├── report_v2.pdf          5-page ACL-style technical report
│   └── documentation.pdf      22-page beginner guide
├── requirements.txt
├── Makefile
└── README.md
```

---

## Trainable components

Four of the five rubric-listed components are fine-tuned here:

| Letter | Component                                           | Where                                                                   |
| ------ | --------------------------------------------------- | ----------------------------------------------------------------------- |
| A      | Retriever (bi-encoder, MNRL fine-tune)              | `models/embeddings.py` + `training/train.py --stage biencoder`          |
| B      | Reranker (cross-encoder)                            | `retrieval/reranker.py`                                                 |
| C      | Generator (Flan-T5 + DoRA adapters)                 | `models/base_model.py`                                                  |
| D      | Preference alignment (DPO on citation-prefix pairs) | `training/train.py --stage dora`                                        |

---

## Technical contributions

Three design choices are specific to this project and not taken from any referenced paper:

1. **Learnable hybrid fusion with manual gradients.** A 2-parameter layer `alpha * cosine + beta * lexical` with `alpha + beta = 1` enforced via softmax, trained with pairwise BPR loss and hand-written gradients (heavy-ball momentum, gradient clipping). PyTorch autograd is intentionally not used — the gradients are written out in source for transparency.
2. **CGRA** (Confidence-Gated Retrieval Augmentation). A composite score combining query-term coverage, post-rerank density, and chunk-length specificity. It decides whether to augment with the web scraper (low confidence) or escalate (very low confidence).
3. **CER** (Correct Escalation Rate). A workflow metric designed for support copilots. On a KB-sufficient evaluation set, `CER = 1 - escalation_rate`; it generalises to mixed sets when the KB-sufficient flag is labelled per query.
