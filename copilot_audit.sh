#!/usr/bin/env bash
# =============================================================================
#  CustomerSupport_Copilot — Full Audit + Auto-Fix Script  (patched 2026-05-12)
#  Usage:  bash copilot_audit.sh [--fix] [--skip-install]
#  Place this file at the ROOT of the CustomerSupport_Copilot repo.
#
#  Patches vs. the original draft:
#   * notebook/colab.ipynb removed from required-files list (Colab support
#     was removed; the canonical entry point is notebook/main.ipynb).
#   * Section 4 tool-name check uses a stricter regex that ignores the
#     backwards-compat aliases (`SearchKBArgs = KBLookupArgs`, etc.).
#   * Section 8 looks for `_should_retrain_fusion` in training/evaluation.py
#     (where it actually lives), not training/agent.py.
#   * Section 9 scans BOTH base_model.py AND quantization.py for
#     device_map="auto"; the quantization.py occurrence in the int8 path is
#     intentional and explicitly skipped.
#   * Section 9 dtype check now greps the `generator:` block, not the
#     first dtype anywhere in the file (which used to hit embeddings.dtype).
#   * Section 12 expected response keys updated to the real AgentResponse
#     schema: final_answer, citations, tool_trace, grounding_score, ticket.
#   * Python interpreter detected: prefers `python3`, falls back to `python`
#     (Windows / Git Bash compatibility).
# =============================================================================
set -uo pipefail
IFS=$'\n\t'

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'
BLU='\033[0;34m'; CYN='\033[0;36m'; NC='\033[0m'
PASS="${GRN}[PASS]${NC}"; FAIL="${RED}[FAIL]${NC}"
WARN="${YEL}[WARN]${NC}"; FIX="${CYN}[FIX ]${NC}"; INFO="${BLU}[INFO]${NC}"

# ── interpreter ──────────────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
  PY=python3
elif command -v python &>/dev/null; then
  PY=python
else
  echo -e "${RED}ERROR: neither python3 nor python on PATH${NC}"
  exit 1
fi

# ── args ──────────────────────────────────────────────────────────────────────
DO_FIX=false
SKIP_INSTALL=false
for arg in "$@"; do
  [[ "$arg" == "--fix" ]]          && DO_FIX=true
  [[ "$arg" == "--skip-install" ]] && SKIP_INSTALL=true
done

ISSUES=0
FIXES=0

log_pass() { echo -e "${PASS} $1"; }
log_fail() { echo -e "${FAIL} $1"; ISSUES=$((ISSUES+1)); }
log_warn() { echo -e "${WARN} $1"; }
log_fix()  { echo -e "${FIX}  $1"; FIXES=$((FIXES+1)); }
log_info() { echo -e "${INFO} $1"; }
section() {
  echo ""
  echo -e "${BLU}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BLU}  $1${NC}"
  echo -e "${BLU}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# =============================================================================
#  0. REPO ROOT CHECK
# =============================================================================
section "0. Repo root check"
if [[ ! -f "requirements.txt" || ! -f "Makefile" ]]; then
  echo -e "${RED}ERROR: Run this script from the root of CustomerSupport_Copilot.${NC}"
  exit 1
fi
log_pass "Running from repo root"

# =============================================================================
#  1. PYTHON VERSION
# =============================================================================
section "1. Python version"
PY_VER=$($PY --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
  log_pass "Python $PY_VER  (need >= 3.10)"
else
  log_fail "Python $PY_VER is too old - need >= 3.10"
fi

# =============================================================================
#  2. REQUIRED FILES EXIST
# =============================================================================
section "2. Required files & folders"
REQUIRED_FILES=(
  "config/config.yaml"
  "config/baseline.yaml"
  "config/full.yaml"
  "data/raw/policy_db.json"
  "models/base_model.py"
  "models/embeddings.py"
  "models/quantization.py"
  "retrieval/retriever.py"
  "retrieval/reranker.py"
  "training/agent.py"
  "training/tools.py"
  "training/router.py"
  "training/memory.py"
  "training/train.py"
  "training/evaluation.py"
  "training/diagnostics.py"
  "training/demo_questions.py"
  "demo/app.py"
  "demo/cli.py"
  "utils/metrics.py"
  "utils/schema.py"
  "utils/config.py"
  "utils/logging.py"
  "utils/device.py"
  "tests/"
  "notebook/main.ipynb"
)
for f in "${REQUIRED_FILES[@]}"; do
  if [[ -e "$f" ]]; then
    log_pass "$f"
  else
    log_fail "MISSING: $f"
  fi
done
# Optional: colab.ipynb should NOT be present anymore
if [[ -e "notebook/colab.ipynb" ]]; then
  log_warn "notebook/colab.ipynb still exists - Colab support was removed; delete it"
fi

# =============================================================================
#  3. PYTHON PACKAGE AVAILABILITY
# =============================================================================
section "3. Python package availability"
if $SKIP_INSTALL; then
  log_info "Skipping install check (--skip-install)"
else
  declare -A PKG_TO_IMPORT=(
    [fastapi]=fastapi
    [uvicorn]=uvicorn
    [pydantic]=pydantic
    [torch]=torch
    [transformers]=transformers
    [sentence-transformers]=sentence_transformers
    [faiss-cpu]=faiss
    [scikit-learn]=sklearn
    [nltk]=nltk
    [rouge-score]=rouge_score
    [beautifulsoup4]=bs4
    [requests]=requests
    [pyyaml]=yaml
    [accelerate]=accelerate
    [peft]=peft
  )
  for pip_name in "${!PKG_TO_IMPORT[@]}"; do
    import_name="${PKG_TO_IMPORT[$pip_name]}"
    if $PY -c "import $import_name" 2>/dev/null; then
      log_pass "import $import_name (pip: $pip_name)"
    else
      log_fail "import $import_name - not installed (pip: $pip_name)"
      if $DO_FIX; then
        log_fix "pip install $pip_name"
        $PY -m pip install --quiet "$pip_name" || true
      fi
    fi
  done
fi

# =============================================================================
#  4. TOOL NAME AUDIT (canonical: KBLookup / PolicyFetch / EscalateIssue)
# =============================================================================
section "4. Tool name audit (reserved names)"
# We allow these legacy names ONLY as right-hand-side identifiers in alias
# lines like `SearchKBArgs = KBLookupArgs`. Anything else is a real hit.
BAD_NAMES=("SearchKB" "GetPolicy" "CreateTicket")
GOOD_NAMES=("KBLookup" "PolicyFetch" "EscalateIssue")
for i in "${!BAD_NAMES[@]}"; do
  bad="${BAD_NAMES[$i]}"
  good="${GOOD_NAMES[$i]}"
  # Count hits that are NOT on an alias line `Old... = New...`
  HITS=$(grep -rEn "\b${bad}[A-Za-z_]*\b" --include="*.py" \
        --exclude-dir=__pycache__ --exclude-dir=.git --exclude-dir=venv . 2>/dev/null \
        | grep -v -E "^[^:]+:[0-9]+:\s*${bad}[A-Za-z_]*\s*=\s*${good}" \
        | grep -v -E "^[^:]+:[0-9]+:\s*${bad,,}[A-Za-z_]*\s*=\s*${good,,}" \
        || true)
  if [[ -z "$HITS" ]]; then
    log_pass "Legacy name '$bad' not used (or only as alias RHS)"
  else
    log_warn "Legacy name '$bad' still referenced - review below:"
    echo "$HITS" | head -5 | sed 's/^/    /'
  fi
done

# =============================================================================
#  5. FASTAPI APP.PY STRUCTURE
# =============================================================================
section "5. FastAPI demo/app.py - structural checks"
APP="demo/app.py"
if [[ -f "$APP" ]]; then
  if grep -qE '@app\.(post|get)\([^)]*["'\'']/query' "$APP"; then
    log_pass "/query route found"
  else
    log_fail "/query endpoint missing"
  fi
  if grep -q "CORSMiddleware" "$APP"; then
    log_pass "CORSMiddleware present"
  else
    log_warn "CORSMiddleware missing - browser UI calls will be blocked"
  fi
  if grep -qE '@app\.(get|post)\([^)]*["'\''](/health|/ping|/status|/)["'\'']' "$APP"; then
    log_pass "/health or / endpoint present"
  else
    log_warn "/health (or /) endpoint missing"
  fi
  # Syntax check
  if $PY -c "import ast; ast.parse(open('$APP').read())" 2>/dev/null; then
    log_pass "demo/app.py parses without syntax errors"
  else
    log_fail "demo/app.py has syntax errors"
    $PY -c "import ast; ast.parse(open('$APP').read())" 2>&1 | tail -3 | sed 's/^/    /'
  fi
else
  log_fail "$APP not found"
fi

# =============================================================================
#  6. FASTAPI LIVE SMOKE TEST
# =============================================================================
section "6. FastAPI live smoke test"
if command -v curl &>/dev/null && curl -sf --max-time 3 http://localhost:8000/health > /dev/null 2>&1; then
  log_pass "Server reachable at http://localhost:8000"
  RESP=$(curl -sf --max-time 30 -X POST http://localhost:8000/query \
    -H 'Content-Type: application/json' \
    -d '{"query":"What documents do I need for retirement benefits?","session_id":"audit-test"}' \
    2>&1 || true)
  if echo "$RESP" | $PY -c "import sys,json; d=json.load(sys.stdin); assert 'final_answer' in d or 'error' in d or 'detail' in d" 2>/dev/null; then
    log_pass "POST /query returned a JSON object with final_answer/error/detail"
  else
    log_warn "POST /query did not return the expected JSON shape"
    echo "    Raw: ${RESP:0:200}"
  fi
else
  log_info "FastAPI server not running - skipping live test (start with: make api)"
fi

# =============================================================================
#  7. RAG PIPELINE CHECKS
# =============================================================================
section "7. RAG pipeline - retriever & reranker"
RETRIEVER="retrieval/retriever.py"
RERANKER="retrieval/reranker.py"
if [[ -f "$RETRIEVER" ]]; then
  if grep -q "BAAI/bge" "$RETRIEVER" && grep -q "all-MiniLM" "$RETRIEVER"; then
    log_fail "MISMATCH: both BGE and all-MiniLM referenced in $RETRIEVER"
  else
    log_pass "Single embedding family in retriever (no obvious mismatch)"
  fi
  if grep -qE "LearnableFusion|alpha|beta" "$RETRIEVER"; then
    log_pass "LearnableFusion (alpha/beta) found"
  else
    log_warn "LearnableFusion not found - hybrid fusion may be off"
  fi
  if grep -qE "faiss\.(read_index|IndexFlatIP|index_factory)" "$RETRIEVER"; then
    log_pass "FAISS index machinery present"
  else
    log_warn "No FAISS index call found in $RETRIEVER"
  fi
else
  log_fail "$RETRIEVER not found"
fi

if [[ -f "$RERANKER" ]]; then
  if grep -qE "CrossEncoder|ms-marco" "$RERANKER"; then
    log_pass "CrossEncoder reranker referenced"
  else
    log_warn "CrossEncoder not found in $RERANKER"
  fi
  if grep -q "list(zip" "$RERANKER" || ! grep -q "zip(" "$RERANKER"; then
    log_pass "Reranker pair construction looks safe"
  else
    log_warn "Reranker may pass a bare zip() iterator to CrossEncoder - verify"
  fi
else
  log_fail "$RERANKER not found"
fi

# Artifacts
if [[ -f "data/processed/chunks.json" ]]; then
  CHUNK_COUNT=$($PY -c "import json; print(len(json.load(open('data/processed/chunks.json'))))" 2>/dev/null || echo "?")
  log_pass "chunks.json exists ($CHUNK_COUNT chunks)"
else
  log_warn "data/processed/chunks.json missing - run: make chunks"
fi
for fbin in data/processed/faiss_index.bin data/processed/faiss_index_ft.bin; do
  [[ -f "$fbin" ]] && log_pass "$fbin exists" || log_warn "$fbin missing"
done
if compgen -G "data/processed/fusion_weights*.json" > /dev/null; then
  log_pass "fusion_weights*.json present"
else
  log_warn "fusion_weights*.json missing - run: make train-fusion"
fi

# =============================================================================
#  8. WEB SCRAPER + COLD-START STUBS
# =============================================================================
section "8. Web scraper + cold-start safety"
AGENT="training/agent.py"
TOOLS="training/tools.py"
EVAL="training/evaluation.py"

# 8a. CGRA scraper is in training/tools.py (web_scraper_tool); agent imports it
if [[ -f "$TOOLS" ]] && grep -qE "BeautifulSoup|web_scraper_tool" "$TOOLS"; then
  log_pass "web_scraper_tool present in training/tools.py"
else
  log_warn "web_scraper_tool not found in training/tools.py"
fi
if [[ -f "$AGENT" ]] && grep -qE "cgra|tau|information_density|web_scraper_tool" "$AGENT"; then
  log_pass "CGRA gate / scraper fallback wired into agent.py"
else
  log_warn "CGRA fallback path not detected in agent.py"
fi
if $PY -c "from bs4 import BeautifulSoup" 2>/dev/null; then
  log_pass "bs4 import works"
else
  log_warn "bs4 not importable (only matters at runtime)"
fi

# 8b. _should_retrain_fusion is DEFINED in training/evaluation.py, not agent.py
if [[ -f "$EVAL" ]] && grep -qE "^def _should_retrain_fusion" "$EVAL"; then
  log_pass "_should_retrain_fusion() is defined in evaluation.py"
elif [[ -f "$EVAL" ]] && grep -q "_should_retrain_fusion" "$EVAL"; then
  log_fail "_should_retrain_fusion is called in evaluation.py but never defined"
else
  log_info "_should_retrain_fusion not referenced (OK if fusion auto-retrain is disabled)"
fi

# =============================================================================
#  9. GENERATOR / DEVICE / DTYPE
# =============================================================================
section "9. Generator - device placement & dtype"
BASE_MODEL="models/base_model.py"
QUANT="models/quantization.py"

# 9a. device_map="auto" should only appear inside the int8/bitsandbytes path,
#     and we need to ignore docstring/comment occurrences. A real code hit is
#     one that is NOT preceded by a `#` and not inside a `"""..."""` block.
_grep_real_devmap() {
  # Print only lines that actually assign device_map="auto" in code,
  # i.e. not inside triple-quoted blocks and not on a comment line.
  $PY - "$1" <<'PYEOF'
import ast, sys, re
path = sys.argv[1]
src = open(path).read()
# Strip all docstrings by parsing the AST
try:
    tree = ast.parse(src)
except SyntaxError:
    print(src)  # fall back to raw text
    sys.exit(0)
# Build a line set covered by docstrings
docstring_lines = set()
for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            n = node.body[0]
            for ln in range(n.lineno, getattr(n, "end_lineno", n.lineno) + 1):
                docstring_lines.add(ln)
for i, line in enumerate(src.splitlines(), 1):
    if i in docstring_lines:
        continue
    stripped = line.lstrip()
    if stripped.startswith("#"):
        continue
    if 'device_map="auto"' in line:
        print(f"{path}:{i}:{line}")
PYEOF
}
HITS_BASE=""
HITS_QUANT=""
[[ -f "$BASE_MODEL" ]] && HITS_BASE=$(_grep_real_devmap "$BASE_MODEL")
[[ -f "$QUANT" ]]      && HITS_QUANT=$(_grep_real_devmap "$QUANT")
if [[ -n "$HITS_BASE" ]]; then
  log_fail 'device_map="auto" found in base_model.py code (not docstring):'
  echo "$HITS_BASE" | sed 's/^/    /'
elif [[ -n "$HITS_QUANT" ]]; then
  # Verify it is inside an int8 / BitsAndBytesConfig block
  if echo "$HITS_QUANT" | head -1 | grep -q ':' && \
     grep -B3 -F "$(echo "$HITS_QUANT" | head -1 | cut -d: -f3-)" "$QUANT" \
       | grep -qE "int8|BitsAndBytes|8bit"; then
    log_pass 'device_map="auto" present only in int8/bitsandbytes path (intentional)'
  else
    log_fail 'device_map="auto" in quantization.py is outside the int8 block:'
    echo "$HITS_QUANT" | sed 's/^/    /'
  fi
else
  log_pass "No stray device_map=auto in code"
fi

# 9b. Explicit .to(device) somewhere in base_model.py
if [[ -f "$BASE_MODEL" ]] && grep -qE "\.to\(device\)|model\.to\(" "$BASE_MODEL"; then
  log_pass "Explicit .to(device) call in base_model.py"
else
  log_warn "No explicit .to(device) - model may stay on CPU"
fi

# 9c. utils/device.py exists and has no module-level DEVICE singleton
if [[ -f "utils/device.py" ]]; then
  if grep -qE "^DEVICE\s*=\s*get_device" utils/device.py; then
    log_fail "utils/device.py has a module-level DEVICE singleton - remove it"
  else
    log_pass "utils/device.py has no module-level DEVICE singleton"
  fi
else
  log_fail "utils/device.py missing"
fi

# 9d. .cuda() shorthand should not appear anywhere
CUDA_SHORTHAND=$(grep -rEn "\.cuda\(\)" --include="*.py" \
  --exclude-dir=__pycache__ --exclude-dir=.git --exclude-dir=venv . 2>/dev/null || true)
if [[ -z "$CUDA_SHORTHAND" ]]; then
  log_pass "No .cuda() shorthand anywhere"
else
  log_fail ".cuda() shorthand found:"
  echo "$CUDA_SHORTHAND" | head -5 | sed 's/^/    /'
fi

# 9e. generator.dtype value
if [[ -f "config/config.yaml" ]]; then
  GEN_DTYPE=$($PY -c "
import yaml
cfg = yaml.safe_load(open('config/config.yaml'))
print(cfg.get('generator', {}).get('dtype', '<missing>'))
" 2>/dev/null)
  case "$GEN_DTYPE" in
    auto)    log_pass "generator.dtype = auto (recommended; resolves at runtime)" ;;
    float32) log_pass "generator.dtype = float32 (safe, slower)" ;;
    float16) log_warn "generator.dtype = float16 - may NaN on some GPUs" ;;
    *)       log_warn "generator.dtype = $GEN_DTYPE (unusual)" ;;
  esac
fi

# =============================================================================
#  10. CONVERSATION MEMORY
# =============================================================================
section "10. Conversation memory"
MEM="training/memory.py"
if [[ -f "$MEM" ]]; then
  if grep -qE "deque|buffer|history" "$MEM"; then
    log_pass "Buffer structure present"
  else
    log_warn "No buffer/history structure found"
  fi
  if grep -qE "max_turns|maxlen|MAX_HISTORY" "$MEM"; then
    log_pass "Buffer size is bounded"
  else
    log_warn "Buffer is unbounded - long sessions will blow context window"
  fi
else
  log_fail "$MEM not found"
fi

# =============================================================================
#  11. OFFLINE PYTEST SUITE
# =============================================================================
section "11. Offline pytest suite"
if $PY -c "import pytest" 2>/dev/null; then
  log_info "Running pytest tests/ -q ..."
  # Note: --timeout requires the pytest-timeout plugin; skip if absent
  TIMEOUT_FLAG=""
  $PY -c "import pytest_timeout" 2>/dev/null && TIMEOUT_FLAG="--timeout=60"
  if $PY -m pytest tests/ -q --tb=short $TIMEOUT_FLAG 2>&1 | tee /tmp/audit_pytest.log | tail -15; then
    log_pass "pytest exited 0"
  else
    log_fail "pytest exited non-zero - see /tmp/audit_pytest.log"
  fi
else
  log_warn "pytest not installed; skipping"
fi

# =============================================================================
#  12. UI / RESPONSE SHAPE
# =============================================================================
section "12. UI / AgentResponse shape"
# Real keys per utils/schema.py::AgentResponse
EXPECTED_KEYS=("final_answer" "citations" "tool_trace" "grounding_score" "ticket")
if [[ -f "utils/schema.py" ]]; then
  for key in "${EXPECTED_KEYS[@]}"; do
    if grep -q "$key" "utils/schema.py"; then
      log_pass "AgentResponse.$key present"
    else
      log_fail "AgentResponse.$key missing in utils/schema.py"
    fi
  done
fi
if [[ -f "demo/app.py" ]] && grep -qE "BaseModel|pydantic|response_model=AgentResponse" "demo/app.py"; then
  log_pass "Pydantic validation present in demo/app.py"
else
  log_warn "No Pydantic validation in demo/app.py"
fi

# =============================================================================
#  13. NLTK DATA
# =============================================================================
section "13. NLTK punkt data"
if $PY -c "import nltk; nltk.data.find('tokenizers/punkt')" 2>/dev/null; then
  log_pass "NLTK punkt found"
else
  log_warn "NLTK punkt data missing"
  if $DO_FIX; then
    $PY -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" \
      && log_fix "Downloaded punkt + punkt_tab"
  fi
fi

# =============================================================================
#  SUMMARY
# =============================================================================
section "SUMMARY"
echo ""
if [[ "$ISSUES" -eq 0 ]]; then
  echo -e "${GRN}All checks passed (no FAIL events).${NC}"
else
  echo -e "${RED}$ISSUES FAIL event(s) found.${NC}"
fi
[[ "$FIXES" -gt 0 ]] && echo -e "${CYN}$FIXES fix(es) applied automatically.${NC}"
echo ""
echo -e "${BLU}Quick commands:${NC}"
echo "  make api           start FastAPI at http://localhost:8000"
echo "  make demo          interactive CLI demo"
echo "  make test          run pytest"
echo "  bash copilot_audit.sh --fix --skip-install   re-run with auto-fix"
echo ""
exit $ISSUES
