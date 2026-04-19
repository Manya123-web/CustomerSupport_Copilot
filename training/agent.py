"""
training/agent.py
-----------------
The core agent orchestrator. Implements:
  • `select_tool()`          keyword-only router (no LLM call)
  • `information_density()`  CGRA confidence score in [0, 1]
  • `grounding_score()`      overlap-based faithfulness metric
  • `generate_answer()`      context assembly + LLM call + extractive fallback
  • `run_agent()`            linear pipeline (NO dead loops) with
                             post-generation grounding re-check and
                             three escalation paths.

Config flags consumed (see config/baseline.yaml vs full.yaml):
  agent.use_tool_loop         — if False, just retrieve→generate, no gate
  agent.use_grounding_recheck — if False, skip the final CER safety net
  cgra.enabled                — if False, skip the ID-gate / WebScraper
  reranker.enabled            — if False, skip cross-encoder rerank
"""
from __future__ import annotations
import math
import re
import statistics
import time
from typing import Callable, Dict, Any, List, Optional

import nltk

from utils.metrics import FULL_STOP, grounding_score as _grounding
from utils.schema import AgentResponse
from training.tools import execute_tool, web_scraper_tool


# Lazy one-time NLTK resource fetch so imports stay cheap
_NLTK_READY = False
def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


# ── Keyword router (NO LLM) ──────────────────────────────────────────────────
TICKET_KEYWORDS = [
    "cannot log", "can't log", "login", "password", "forgot password",
    "account locked", "locked out", "not working", "broken",
    "unable to access", "error", "issue with my account", "my account",
    "sign in", "reset", "access denied",
]
POLICY_KEYWORDS = ["policy", "section", "regulation", "rule", "statute"]


def select_tool(query: str) -> str:
    q = query.lower()
    if any(k in q for k in TICKET_KEYWORDS):
        return "CreateTicket"
    if any(k in q for k in POLICY_KEYWORDS):
        return "GetPolicy"
    return "SearchKB"


# ── Policy-topic to section_id mapper ────────────────────────────────────────
# The policy DB (data/raw/policy_db.json) uses keys section_1..section_N.
# A raw user query never matches these verbatim, so we maintain a small
# keyword->section dictionary here. The mapping is LOADED FROM CONFIG if
# a `policy_topic_map` key is present, otherwise falls back to this default.
_DEFAULT_POLICY_TOPIC_MAP = {
    "section_1": ["retirement", "retire", "work credits", "full retirement age",
                   "early retirement", "age 62", "age 67"],
    "section_2": ["ssdi", "disability insurance", "disability benefit",
                   "substantial gainful activity", "medically determinable"],
    "section_3": ["survivor", "survivors", "widow", "widower", "deceased",
                   "surviving spouse", "minor children"],
    "section_4": ["ssi", "supplemental security income", "needs-based",
                   "limited income", "general tax revenue"],
    "section_5": ["medicare", "part a", "part b", "initial enrollment",
                   "turning 65", "age 65"],
}

# Allow explicit references like "section 3", "sec 3", "section_3"
_SECTION_REF = re.compile(r"\bsec(?:tion)?[\s_]*([0-9]+)\b", re.IGNORECASE)


def _map_query_to_policy_section(query: str,
                                  topic_map: Optional[Dict[str, List[str]]] = None
                                  ) -> Optional[str]:
    """
    Given a user query like "What's the rule for disability benefits?",
    return a policy-DB key like "section_2". Returns None if nothing matches.

    Matching strategy (in order):
      1. Explicit reference ("section 3" -> "section_3")
      2. Topic-keyword match against the longest matching topic term
    """
    q = query.lower()

    # 1) explicit reference
    m = _SECTION_REF.search(q)
    if m:
        candidate = f"section_{m.group(1)}"
        return candidate

    # 2) topic-keyword match — prefer longest, most specific matches
    topic_map = topic_map or _DEFAULT_POLICY_TOPIC_MAP
    best_section: Optional[str] = None
    best_len = 0
    for section_id, keywords in topic_map.items():
        for kw in keywords:
            if kw in q and len(kw) > best_len:
                best_section = section_id
                best_len = len(kw)
    return best_section


# ── CGRA information-density gate ────────────────────────────────────────────
def information_density(query: str, chunks: List[Dict[str, Any]],
                         a_id: float = 0.4, b_id: float = 0.4,
                         g_id: float = 0.2) -> float:
    if not chunks:
        return 0.0
    q_words  = set(re.findall(r"\w+", query.lower())) - FULL_STOP
    all_wrds = set(w for c in chunks for w in re.findall(r"\w+", c["text"].lower()))
    coverage = len(q_words & all_wrds) / len(q_words) if q_words else 0.0

    raw = []
    for c in chunks:
        if "ce_score" in c and c["ce_score"] is not None:
            raw.append(1.0 / (1.0 + math.exp(-c["ce_score"])))
        elif "cos_score" in c and c["cos_score"] is not None:
            raw.append(max(0.0, min(1.0, c["cos_score"])))
        else:
            raw.append(max(0.0, min(1.0, c.get("score", 0.0))))
    density = sum(raw) / len(raw) if raw else 0.0

    lengths = [len(c["text"].split()) for c in chunks]
    specificity = (1.0 - min(statistics.stdev(lengths) / 150, 1.0)
                   if len(lengths) > 1 else 0.8)
    return a_id * coverage + b_id * density + g_id * specificity


# ── Extractive fallback sentence selection ───────────────────────────────────
def _best_sentence(query: str, context: str) -> str:
    _ensure_nltk()
    try:
        sents = nltk.sent_tokenize(context.strip())
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", context.strip())
    if not sents:
        return context[:300]
    qw = set(re.findall(r"\w+", query.lower()))
    return max(sents, key=lambda s: len(qw & set(re.findall(r"\w+", s.lower()))))


# ── Context assembly + generation ────────────────────────────────────────────
MAX_CONTEXT_WORDS = 380


def generate_answer(query: str, chunks: List[Dict[str, Any]],
                    llm_fn: Callable,
                    grounding_threshold: float = 0.08,
                    history_prefix: str = "") -> tuple:
    context, citations, seen = "", [], set()
    for c in chunks:
        if c["doc_id"] in seen:
            continue
        seen.add(c["doc_id"])
        text = re.sub(r"\[\d+\]", "", c["text"]).strip()
        if len(text.split()) < 10:
            continue
        candidate = context + text + "\n\n"
        if len(candidate.split()) > MAX_CONTEXT_WORDS:
            remaining = MAX_CONTEXT_WORDS - len(context.split())
            if remaining > 20:
                context += " ".join(text.split()[:remaining]) + "\n\n"
                citations.append(c["chunk_id"])
            break
        context += text + "\n\n"
        citations.append(c["chunk_id"])

    if len(context.strip()) < 50:
        return "Sorry, no relevant information found.", [], 0.0, context

    first_doc = citations[0].split("__")[0] if citations else "KB"
    # `history_prefix` is empty for turn 1; for turn 2+ it contains the
    # "Previous conversation: ..." block from ConversationMemory
    prompt = (
        "You are a customer support agent.\n"
        f"Answer ONLY using the context. Start with [doc_id: {first_doc}].\n"
        "Be specific and complete.\n\n"
        f"{history_prefix}"
        f"Context:\n{context.strip()}\n\n"
        f"Question: {query}\n\nAnswer [cite first]:"
    )
    resp   = llm_fn(prompt)
    answer = resp[0]["generated_text"].strip()
    if len(answer) < 5:
        answer = "Sorry, could not generate a meaningful answer."

    gs = _grounding(answer, context)
    if gs < grounding_threshold:
        answer = _best_sentence(query, context)
        gs     = _grounding(answer, context)
    return answer, citations, gs, context


# ── Full agent loop ──────────────────────────────────────────────────────────
def run_agent(query: str,
              retriever,
              llm_fn: Callable,
              reranker=None,
              cfg: Optional[dict] = None,
              learned_router=None,
              memory=None) -> AgentResponse:
    """
    Single linear pipeline (no dead loops). Emits an AgentResponse.

    Parameters
    ----------
    learned_router : LearnedRouter or None
        If provided AND cfg["agent"]["router"] == "learned", use it
        instead of the deterministic keyword matcher.
    memory : ConversationMemory or None
        Used for (a) pronoun resolution in retrieval, (b) history prefix
        in generation, (c) per-turn recording.
    """
    cfg = cfg or {}

    # Resolve pronoun references using conversation memory (if present)
    retrieval_query = query
    if memory is not None and len(memory) > 0:
        retrieval_query = memory.resolve_reference(query)
        if retrieval_query != query:
            # keep original in prompt, but retrieve on expanded query
            pass
    use_tool_loop         = cfg.get("agent", {}).get("use_tool_loop", True)
    use_grounding_recheck = cfg.get("agent", {}).get("use_grounding_recheck", True)
    cgra_enabled          = cfg.get("cgra", {}).get("enabled", True)
    rerank_enabled        = cfg.get("reranker", {}).get("enabled", True)
    tau                   = cfg.get("cgra", {}).get("tau", 0.45)
    tau_min               = cfg.get("cgra", {}).get("tau_min", 0.20)
    a_id                  = cfg.get("cgra", {}).get("a_id", 0.4)
    b_id                  = cfg.get("cgra", {}).get("b_id", 0.4)
    g_id                  = cfg.get("cgra", {}).get("g_id", 0.2)
    grounding_threshold   = cfg.get("grounding", {}).get("threshold", 0.08)
    top_k                 = cfg.get("agent", {}).get("top_k", 5)
    synth_top_k           = cfg.get("agent", {}).get("synthesis_top_k", 8)

    t0 = time.perf_counter()
    trace: List[Dict[str, Any]] = []
    q_lower = query.lower()

    # ── baseline short-circuit: straight retrieve → generate, no tools ───────
    if not use_tool_loop:
        chunks = retriever.retrieve(query, top_k=top_k)
        if rerank_enabled and reranker is not None:
            chunks = reranker.rerank(query, chunks, top_n=top_k)
        history_prefix = memory.history_prompt() if memory is not None else ""
        answer, cites, gs, ctx = generate_answer(query, chunks, llm_fn,
                                                  grounding_threshold,
                                                  history_prefix=history_prefix)
        if memory is not None:
            memory.add_turn(query, answer, grounding=gs)
        return AgentResponse(
            final_answer=answer, citations=cites,
            grounding_score=gs, id_score=0.0,
            tool_trace=[{"step": 1, "tool": "SearchKB_baseline"}],
            latency_ms=(time.perf_counter() - t0) * 1000)

    # ── full tool loop ───────────────────────────────────────────────────────
    synth_kw = ["documents","list","steps","how to","what do i need",
                "requirements","apply","process","how do","how can",
                "how long","what is","explain"]
    if any(k in q_lower for k in synth_kw):
        top_k = max(top_k, synth_top_k)

    # (1) routing — keyword by default; learned if configured AND provided
    router_kind = cfg.get("agent", {}).get("router", "keyword")
    if router_kind == "learned" and learned_router is not None:
        tool = learned_router.predict(query)
    else:
        tool = select_tool(query)
    step = {"step": 1, "tool": tool, "router": router_kind}

    # (2) build arguments
    if tool == "CreateTicket":
        cat = ("account"  if any(k in q_lower for k in ["login","password","sign in","access","locked"])
               else "payments" if any(k in q_lower for k in ["payment","benefit","deposit"])
               else "general")
        sev = "high" if any(k in q_lower for k in ["urgent","broken","error","cannot","can't"]) else "medium"
        call = {"tool": "CreateTicket",
                "arguments": {"summary": f"User query: {query}",
                              "category": cat, "severity": sev}}
    elif tool == "GetPolicy":
        # Map the query to a specific policy section. The naive version
        # passed the raw query as section_id, which never matched
        # policy_db.json keys ("section_1", "section_2", …) and always
        # silently fell back to KB retrieval. We now do two things:
        #   1) surface-level topic keywords -> known section id
        #   2) fall through to `query` field so the tool dispatcher
        #      can still use the KB-fallback path if no topic matches
        section_id = _map_query_to_policy_section(query) or query
        call = {"tool": "GetPolicy",
                "arguments": {"section_id": section_id, "query": query}}
    else:
        # Use the memory-expanded query if a reference was resolved
        call = {"tool": "SearchKB",
                "arguments": {"query": retrieval_query, "top_k": top_k}}

    # (3) execute (schema-validated)
    result = execute_tool(call, retriever=retriever)
    step["result_preview"] = str(result)[:150]
    trace.append(step)

    # (3a) direct CreateTicket branch — done
    if tool == "CreateTicket":
        ticket_answer = (f"Issue escalated.\nTicket ID: {result['ticket_id']}\n"
                          f"Category: {result.get('category')}  "
                          f"Severity: {result.get('severity')}\n"
                          f"A representative will follow up.")
        if memory is not None:
            memory.add_turn(query, ticket_answer, has_ticket=True)
        return AgentResponse(
            final_answer=ticket_answer,
            ticket=result, tool_trace=trace,
            latency_ms=(time.perf_counter() - t0) * 1000)

    # (4) collect chunks (KB or policy-text)
    chunks = result.get("results", [])
    if not chunks and result.get("text"):
        chunks = [{"chunk_id": f"policy__{result.get('section_id','')}",
                   "doc_id":   f"policy__{result.get('section_id','')}",
                   "text":     result["text"], "score": 1.0}]
    # Make it visible in the trace when GetPolicy actually fell back to KB
    # retrieval — the auditor flagged that this was hidden before.
    if result.get("tool") == "GetPolicy_KB_fallback":
        step["policy_fallback"] = True
        step["policy_note"] = result.get("note", "")

    # (5) rerank FIRST so the CGRA gate can read post-rerank ce_score
    if rerank_enabled and reranker is not None and chunks:
        chunks = reranker.rerank(query, chunks, top_n=max(5, top_k))

    # (6) CGRA gate — now reads ce_score as the design intended
    id_score = 0.0
    if cgra_enabled:
        id_score = information_density(query, chunks, a_id, b_id, g_id)
        step["id_score"] = round(id_score, 4)
        if id_score < tau and tool == "SearchKB":
            web = web_scraper_tool(query)
            if web:
                # Web chunks don't have ce_score; rerank the combined set so
                # everything feeding the second gate is on the same scale
                chunks = chunks + web
                if rerank_enabled and reranker is not None:
                    chunks = reranker.rerank(query, chunks, top_n=max(5, top_k))
                id_aug = information_density(query, chunks, a_id, b_id, g_id)
                step["id_aug"] = round(id_aug, 4)
                if id_aug < tau_min:
                    ticket = execute_tool(
                        {"tool": "CreateTicket",
                         "arguments": {"summary": f"Low-confidence: {query}",
                                       "category": "general", "severity": "medium"}})
                    return AgentResponse(
                        final_answer=f"No confident answer.\nTicket: {ticket['ticket_id']}",
                        ticket=ticket, id_score=id_score, tool_trace=trace,
                        latency_ms=(time.perf_counter() - t0) * 1000)

    if not chunks:
        ticket = execute_tool(
            {"tool": "CreateTicket",
             "arguments": {"summary": f"No KB answer: {query}",
                           "category": "general", "severity": "medium"}})
        return AgentResponse(
            final_answer=f"No answer. Ticket: {ticket['ticket_id']}",
            ticket=ticket, id_score=id_score, tool_trace=trace,
            latency_ms=(time.perf_counter() - t0) * 1000)

    # (7) generate (chunks already reranked above)
    # Trim to top-5 for the generator's context budget
    chunks = chunks[:5]

    history_prefix = memory.history_prompt() if memory is not None else ""
    answer, cites, gs, ctx = generate_answer(query, chunks, llm_fn,
                                              grounding_threshold,
                                              history_prefix=history_prefix)
    step["grounding_score"] = round(gs, 4)

    # (8) POST-generation grounding re-check (CER safety net)
    if use_grounding_recheck and gs < grounding_threshold:
        ticket = execute_tool(
            {"tool": "CreateTicket",
             "arguments": {"summary": f"Ungrounded answer: {query}",
                           "category": "general", "severity": "high"}})
        escalate_text = (f"Answer could not be grounded in the knowledge base.\n"
                          f"Ticket: {ticket['ticket_id']} (a human will follow up).")
        if memory is not None:
            memory.add_turn(query, escalate_text, grounding=gs, has_ticket=True)
        return AgentResponse(
            final_answer=escalate_text,
            ticket=ticket, citations=cites, grounding_score=gs,
            id_score=id_score, tool_trace=trace,
            latency_ms=(time.perf_counter() - t0) * 1000)

    if memory is not None:
        memory.add_turn(query, answer, grounding=gs)

    return AgentResponse(
        final_answer=answer, citations=cites,
        grounding_score=gs, id_score=id_score,
        tool_trace=trace,
        latency_ms=(time.perf_counter() - t0) * 1000)
