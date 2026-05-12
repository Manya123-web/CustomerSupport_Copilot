# CLAUDE.md — CustomerSupport Copilot

> Place this file at the **root** of the repository (`CustomerSupport_Copilot/CLAUDE.md`).
> Claude Code reads it automatically on every session startup.

---

## 1. Project identity

| Field | Value |
|---|---|
| **Name** | Customer Support Copilot |
| **Purpose** | Document-grounded RAG system: answers KB questions, opens support tickets, and escalates to humans when unsure |
| **Language** | Python 3.10+ |
| **Primary entry points** | `Makefile` targets — always prefer `make <target>` over running scripts directly |
| **Notebook** | `notebook/main.ipynb` (local chunking + eval) |
| **Serving** | FastAPI at `demo/app.py` → `POST /query` |
| **Dataset** | MultiDoc2Dial (Social Security / government benefits), drop at `data/raw/multidoc2dial_raw.json` |

---

## 2. Folder structure (canonical)

```
CustomerSupport_Copilot/
├── config/
│   ├── config.yaml          # Shared hyperparameters — loaded by every run via utils/config.py
│   ├── baseline.yaml        # Frozen encoder, no reranker, no DoRA, no CGRA
│   └── full.yaml            # Everything enabled (extends config.yaml)
│
├── data/
│   ├── raw/
│   │   ├── multidoc2dial_raw.json   # ← dataset goes here (not committed)
│   │   └── policy_db.json           # Static policy DB (committed)
│   ├── processed/           # Generated: chunks.json, faiss_*.bin, fusion_weights.json
│   └── splits/              # Generated: train.json + test.json (80/20 doc-level split)
│
├── models/
│   ├── base_model.py        # Flan-T5-large loader with strict DoRA attach + auto device (cuda/cpu)
│   ├── embeddings.py        # Bi-encoder load + MNRL fine-tune
│   └── quantization.py      # fp16 / fp32 / optional int8 helpers
│
├── retrieval/
│   ├── retriever.py         # FAISS index + LearnableFusion (manual gradient descent on α, β)
│   └── reranker.py          # ms-marco-MiniLM-L-6-v2 cross-encoder reranker
│
├── training/
│   ├── train.py             # CLI: --stage biencoder | fusion | dora | router | all
│   ├── evaluation.py        # CLI: --config <yaml> | --compare
│   ├── agent.py             # CGRA gate + router + linear agent loop + grounding check
│   ├── tools.py             # Pydantic-validated tools: KBLookup / PolicyFetch / EscalateIssue
│   ├── router.py            # Learned logistic-regression router (opt-in)
│   ├── memory.py            # Bounded per-session conversation buffer
│   ├── diagnostics.py       # Overfitting / underfitting / leakage checks
│   └── demo_questions.py    # 30 curated demo questions across 4 behaviour categories
│
├── utils/
│   ├── metrics.py           # P@K, R@K, MRR, BLEU, ROUGE-L, grounding, CER, CFS
│   ├── schema.py            # Pydantic ToolCall + AgentResponse schemas
│   ├── config.py            # YAML loader with `extends:` inheritance
│   └── logging.py           # JSON-line run logger
│
├── demo/
│   ├── app.py               # FastAPI server — POST /query, per-session memory
│   └── cli.py               # Interactive terminal demo
│
├── notebook/
│   └── main.ipynb           # Local thin orchestrator (chunking + eval)
│
├── tests/                   # pytest suite (offline, runs in seconds, no GPU needed)
├── experiments/results/     # Timestamped JSON metric dumps from each run
├── report/                  # LaTeX/PDF project report (do not auto-edit)
├── requirements.txt         # All Python dependencies, pinned
├── Makefile                 # Top-level entry points — prefer this over raw python calls
└── CLAUDE.md                # ← this file
```

---

## 3. Tool names (important — professor-approved names)

The three agent tools in `training/tools.py` use these **exact class/function names**:

| Tool | Name in code | What it does |
|---|---|---|
| Knowledge base search | `KBLookup` | Retrieves relevant chunks from FAISS index |
| Policy fetch | `PolicyFetch` | Reads entries from `data/raw/policy_db.json` |
| Ticket / escalation | `EscalateIssue` | Creates a support ticket or escalates to human |

> **Never** rename these back to `SearchKB`, `GetPolicy`, or `CreateTicket`. Those names are retired.
> All Pydantic schemas in `utils/schema.py` must reference the names above.

---

## 4. Key commands (always use `make`)

```bash
# First-time setup
make install          # pip install -r requirements.txt + NLTK data

# Data preparation (run once after install or dataset change)
make chunks           # runs notebook/main.ipynb headless → writes processed/ and splits/

# Training (each stage is independent; run in order for a full pipeline)
make train-biencoder  # fine-tunes MiniLM dense retriever (~5 min on T4)
make train-fusion     # fits α/β hybrid fusion weights via BPR (<1 min)
make train-dora       # DoRA adapters on Flan-T5-large + DPO loss (~15–20 min on T4)
make train-router     # trains logistic-regression router (optional, <1 min)
make train            # all four stages in order

# Evaluation (read-only — never modifies checkpoints)
make eval-baseline    # baseline config (no reranker, no DoRA, no CGRA)
make eval-full        # full config (everything enabled)
make compare          # prints baseline-vs-full delta table → writes comparison_latest.json
make eval             # all three in order

# Testing
make test             # runs pytest suite (offline, no GPU, a few seconds)

# Interactive use
make demo             # terminal REPL — type a query, see answer + tool trace
make api              # FastAPI server at http://localhost:8000

# One-off eval override (when checkpoints missing)
COPILOT_ALLOW_EVAL_TRAIN=1 make eval-full
```

---

## 5. Architecture & data-flow

```
User query
    │
    ▼
Router (keyword matcher, default)
  ├─ account/action intent ──► EscalateIssue (skip retrieval)
  └─ informational intent
        │
        ▼
   KBLookup
     ├─ Bi-encoder (MiniLM) → FAISS top-K  ─┐
     └─ Jaccard lexical match               ─┤ LearnableFusion (α·dense + β·lexical)
                                             │
                                             ▼
                                    Cross-encoder reranker
                                    (ms-marco-MiniLM-L-6-v2)
                                             │
                                             ▼
                                    CGRA confidence gate
                                      ├─ HIGH → Flan-T5-large (DoRA) generation
                                      │           └─ Grounding check
                                      │               ├─ GROUNDED → return answer
                                      │               └─ UNGROUNDED → EscalateIssue
                                      └─ LOW  → WebScraper fallback → re-check → EscalateIssue
```

**Conversation memory** (`training/memory.py`): bounded per-session buffer, resolves pronouns, injects dialogue history into generation prompt. Keyed by `session_id` in the FastAPI layer.

---

## 6. Config system

Configs use `extends:` inheritance via `utils/config.py`:

- `config.yaml` is the base (shared hyperparameters for all runs).
- `baseline.yaml` and `full.yaml` both `extends: config/config.yaml` and override specific keys.
- **Never hardcode hyperparameters in Python files** — always read from config.
- Generator dtype lives at `generator.dtype` in `config.yaml` (default `"auto"` — fp16 on CUDA, fp32 on CPU/MPS; set `"float32"` explicitly if NaN loss occurs).

---

## 7. GPU & hardware notes

| Environment | Status |
|---|---|
| Any NVIDIA GPU ≥ 6 GB VRAM (local) | **Primary training target** — works in fp16 |
| Apple Silicon (M-series, MPS) | Supported for inference; training works but is slower than CUDA |
| CPU-only (e.g. Intel i5, no dedicated GPU) | **Fully supported for inference and demo.** Training is slow but works — expect 5–10× longer times than a recent GPU |
| Integrated GPU (Intel UHD, < 1 GB VRAM) | Not usable for PyTorch CUDA — treated as CPU-only |
| Google Colab T4 (free tier) | Optional, for users without a local GPU — sufficient for all training + eval stages |

### How device selection works

`models/base_model.py` auto-detects the best available device via the shared `utils.device.get_device()` helper, which picks CUDA (NVIDIA/AMD) → MPS (Apple Silicon) → CPU in that order, picking the CUDA device with the most free VRAM on multi-GPU machines:

```python
from utils.device import get_device
device = get_device()      # runtime detection, called inside the loader
model = model.to(device)
```

At startup it prints a `[base_model] device=cpu` (or `cuda:0`, `mps`, ...) log line — always check this first if behaviour seems slow or wrong.

### Running locally on CPU (Intel i5 / no GPU)

- **Inference / demo / tests:** run normally with `make demo`, `make api`, `make test` — all work on CPU.
- **Training:** works but is slow. Recommended order to minimise wait time:
  1. `make train-fusion` — fast (<2 min even on CPU)
  2. `make train-router` — fast (<2 min on CPU)
  3. `make train-biencoder` — moderate (~20–40 min on CPU)
  4. `make train-dora` — slow (hours on CPU; **use a CUDA GPU — or Colab T4 if you don't have one locally — for this stage**)
- Set `generator.dtype: "float32"` in `config/config.yaml` when running on CPU (fp16 is GPU-optimised).

### Common device errors

- `RuntimeError: Expected all tensors on the same device` — do **not** use `device_map="auto"`; the explicit `.to(device)` in `base_model.py` is the correct fix.
- `AssertionError: Torch not compiled with CUDA` — you are on a CPU-only install; this is expected. The code falls back to CPU automatically.

---

## 8. Evaluation rules

- Evaluation (`training/evaluation.py`) is **read-only by design** — it never writes new checkpoints.
- If a required artifact is missing (fusion weights, router checkpoint), eval will stop with an actionable error. Run the corresponding `make train-*` step first.
- Result files: `experiments/results/<config>_<timestamp>.json` — do not delete; `compare` reads the latest pair.
- The held-out test set in `data/splits/test.json` must **never** be used during training. Leakage check is in `training/diagnostics.py`.

**Key metrics to track:**

| Metric | Meaning |
|---|---|
| P@5, R@5, MRR | Retrieval quality |
| BLEU, ROUGE-L | Generation n-gram overlap |
| grounding | Fraction of answer tokens present in retrieved context |
| CER | Correct Escalation Rate (project-specific) |
| escalation_rate | Fraction of queries that opened a ticket |
| latency_p50_ms / p95_ms | Per-query inference latency |

**Reference deltas (baseline → full):** P@5 +0.14, grounding +0.14, CER +0.30, escalation_rate −0.30.

---

## 9. Custom metrics (know what they mean)

- **CER (Correct Escalation Rate):** On a KB-sufficient query set, `CER = 1 − escalation_rate`. High CER means the system answers when it should, and escalates when it must.
- **CFS (Citation Fidelity Score):** Measures whether generated answers include the expected source citation prefix. Defined in `utils/metrics.py`.
- **Grounding:** Lexical token overlap between generated answer and retrieved context — *not* NLI entailment. This is a known limitation (see §11).

---

## 10. Testing

```bash
make test   # runs the full pytest suite
```

The suite covers:
- `utils/metrics.py` — all metric functions
- `training/agent.py` — router dispatch logic
- `utils/schema.py` — Pydantic `ToolCall` / `AgentResponse` validation
- `data/raw/policy_db.json` — policy DB structure
- Policy mapper correctness
- `training/memory.py` — conversation buffer
- `training/demo_questions.py` — demo question manifest (30 questions, 4 categories)

Tests are offline and require no GPU. They should pass in a few seconds. **Never merge code that breaks the test suite.**

---

## 11. Known limitations (do not try to "fix" without discussion)

- Grounding score is **lexical token overlap**, not NLI entailment. This is intentional (speed). A future improvement is an NLI classifier at `agent.py::_grounding`.
- No continuous learning from user interactions — training is fully offline; serving is read-only inference.
- The 20-query eval set and 30-question demo set are illustrative, not a full benchmark.
- No real validation split — only train/test. Early stopping for bi-encoder is a future improvement.

---

## 12. Common pitfalls & fixes

| Problem | Fix |
|---|---|
| `Fusion weights not found` during eval | `make train-fusion` |
| `Learned router checkpoint not found` | `make train-router` or set `agent.router_autotrain_in_eval: true` in config |
| DoRA loss is NaN on a GPU | Set `generator.dtype: "float32"` in `config/config.yaml` |
| `RuntimeError: Expected all tensors on same device` | Do not use `device_map="auto"`; `base_model.py` uses an explicit `.to(get_device())` |
| `LookupError` for NLTK `punkt_tab` | `pip install --upgrade "nltk>=3.9"` then `python -c "import nltk; nltk.download('punkt_tab')"` |
| GPU shows zero load during encoding | Expected on CPU-only machines. If you have a CUDA GPU, confirm `torch.cuda.is_available()` is `True` and PyTorch was installed with CUDA support |

---

## 13. Code style & contribution rules

- **Python 3.10+** — use `match`/`case`, `X | Y` union types, `dataclasses` where appropriate.
- All public functions and classes must have **docstrings**.
- All tool schemas go through **Pydantic v2** — no raw dicts as tool call inputs.
- Hyperparameters live in **YAML configs**, not in Python source files.
- Log with `utils/logging.py` (JSON-line format) — do not use bare `print()` in library code.
- `demo/cli.py` and `demo/app.py` are the only places where user-facing `print()` / `logging.info()` is acceptable.
- Run `make test` before every commit. All tests must pass.
- Do not commit generated artifacts (`data/processed/`, `data/splits/`, `models/checkpoints/`, `experiments/results/`) unless specifically asked.

---

## 14. API reference (FastAPI)

```
POST http://localhost:8000/query
Content-Type: application/json

{
  "query": "What documents do I need to apply for retirement benefits?",
  "session_id": "user-42"
}
```

- `session_id` is required for conversation memory and pronoun resolution.
- Reusing the same `session_id` on follow-up queries enables multi-turn dialogue context.
- Response shape is `AgentResponse` from `utils/schema.py`.

---

## 15. What NOT to do

- Do **not** run `python training/train.py` directly — always use `make train-*`.
- Do **not** edit `data/splits/test.json` — it is the held-out test set.
- Do **not** call tools by the old names (`SearchKB`, `GetPolicy`, `CreateTicket`).
- Do **not** add `device_map="auto"` to model loading in `base_model.py` — use the explicit `model.to(device)` pattern instead.
- Do **not** hardcode dataset paths — use `utils/config.py` to read from `config.yaml`.
- Do **not** commit secrets, API keys, or HuggingFace tokens to the repo.
- Do **not** modify `report/` files unless explicitly asked — it is the submitted academic report.
