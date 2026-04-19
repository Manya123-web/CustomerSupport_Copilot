"""
utils/metrics.py
----------------
All evaluation metrics in one place so baseline and full runs share identical
scoring code. Nothing in this file imports a model — it's pure numeric.

Metric groups
-------------
Retrieval     : precision_at_k, recall_at_k, mrr
Generation    : bleu, rouge_l
Grounding     : grounding_score (non-stop-word overlap with context)
Workflow      : escalation_rate, correct_escalation_rate (CER)

`CER` is one of the "own novel evaluation criteria" claimed for the project:
on a KB-sufficient evaluation set every escalation is an error, so
CER = 1 - escalation_rate. It is independent of BLEU/ROUGE and captures a
production-relevant behaviour (did the system wrongly punt to a human?).
"""

from __future__ import annotations
import re
import math
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ── Unified stopword list (used by grounding + lexical scorers) ──────────────
_LOCAL_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "but", "or", "nor", "not", "only", "same", "than", "too",
    "very", "just", "if", "this", "that", "it", "i", "you", "we",
    "they", "he", "she", "which", "who", "each", "both",
}
FULL_STOP = _LOCAL_STOP | set(ENGLISH_STOP_WORDS)

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower())) - FULL_STOP


# ── Retrieval metrics ────────────────────────────────────────────────────────
def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable, k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant_ids)
    return sum(1 for d in retrieved_ids[:k] if d in rel) / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable, k: int) -> float:
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    return sum(1 for d in retrieved_ids[:k] if d in rel) / len(rel)


def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable) -> float:
    rel = set(relevant_ids)
    for rank, d in enumerate(retrieved_ids, 1):
        if d in rel:
            return 1.0 / rank
    return 0.0


# ── Generation metrics ───────────────────────────────────────────────────────
_SMOOTH = SmoothingFunction().method1
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def bleu(hypothesis: str, reference: str) -> float:
    h, r = hypothesis.split(), reference.split()
    if not h or not r:
        return 0.0
    return sentence_bleu([r], h, smoothing_function=_SMOOTH)


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    return _ROUGE.score(reference, hypothesis)["rougeL"].fmeasure


# ── Grounding: the one required "evidence-use" metric ────────────────────────
def grounding_score(answer: str, context: str) -> float:
    """
    Fraction of NON-STOP answer words that appear in the retrieved context.
    Range [0, 1]; higher means the answer is more faithful to the evidence.
    """
    a = _tokens(answer)
    if not a:
        return 0.0
    c = _tokens(context)
    return len(a & c) / len(a)


# ── Workflow metrics (novelty) ───────────────────────────────────────────────
def escalation_rate(agent_outputs: Sequence[dict]) -> float:
    """Fraction of agent outputs that opened a ticket instead of answering."""
    if not agent_outputs:
        return 0.0
    n = sum(1 for o in agent_outputs if o.get("ticket"))
    return n / len(agent_outputs)


def correct_escalation_rate(agent_outputs: Sequence[dict],
                            kb_sufficient: Sequence[bool] | None = None) -> float:
    """
    CER = fraction of queries handled correctly w.r.t. the escalation decision.

    If `kb_sufficient` is None we assume every eval query is KB-sufficient
    (as is the case for the 100-query eval set), in which case
        CER = 1 - escalation_rate
    If it is supplied, we count per-query correctness:
        correct = (kb_sufficient AND not escalated) OR (NOT kb_sufficient AND escalated)
    """
    if not agent_outputs:
        return 0.0
    if kb_sufficient is None:
        return 1.0 - escalation_rate(agent_outputs)
    correct = 0
    for out, suff in zip(agent_outputs, kb_sufficient):
        esc = bool(out.get("ticket"))
        if (suff and not esc) or ((not suff) and esc):
            correct += 1
    return correct / len(agent_outputs)


# ── One-shot aggregator used by training/evaluation.py ───────────────────────
def aggregate_metrics(records: Sequence[dict]) -> dict:
    """
    records[i] must contain:
        retrieved_ids, relevant_ids, answer, reference, context,
        ticket (bool-ish), kb_sufficient (optional bool).
    Returns a flat dict of averaged metrics.
    """
    if not records:
        return {}
    ks = [5, 10, 15, 20]
    out = {f"P@{k}": 0.0 for k in ks}
    out.update({f"R@{k}": 0.0 for k in ks})
    out["MRR"] = 0.0
    out["BLEU"] = 0.0
    out["ROUGE-L"] = 0.0
    out["grounding"] = 0.0

    for r in records:
        for k in ks:
            out[f"P@{k}"] += precision_at_k(r["retrieved_ids"], r["relevant_ids"], k)
            out[f"R@{k}"] += recall_at_k(r["retrieved_ids"], r["relevant_ids"], k)
        out["MRR"] += mrr(r["retrieved_ids"], r["relevant_ids"])
        out["BLEU"] += bleu(r["answer"], r["reference"])
        out["ROUGE-L"] += rouge_l(r["answer"], r["reference"])
        out["grounding"] += grounding_score(r["answer"], r["context"])

    n = len(records)
    for k in list(out):
        out[k] /= n

    kb_suff = [r.get("kb_sufficient", True) for r in records]
    out["escalation_rate"] = escalation_rate(records)
    out["CER"] = correct_escalation_rate(records, kb_suff)
    out["n"] = n
    return out
