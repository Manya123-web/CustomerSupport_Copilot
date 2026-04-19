"""
training/diagnostics.py
-----------------------
Single source of truth for overfitting / underfitting / leakage checks.

Prints a human-readable table showing, for each trainable component:
  * a TRAIN score (the metric on examples the model did see)
  * a TEST  score (the metric on examples the model did NOT see)
  * a DELTA (train - test)
  * a verdict (OK / UNDERFIT / OVERFIT / LEAKAGE-RISK)

Call from a notebook:
    from training.diagnostics import run_diagnostics
    run_diagnostics(cfg_path="config/full.yaml")
"""
from __future__ import annotations
import json
import os
import random
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
def _doc_disjoint(train_docs: set, test_docs: set) -> Tuple[bool, int]:
    """True iff train and test share zero documents. Also returns overlap size."""
    common = train_docs & test_docs
    return (len(common) == 0), len(common)


def _eval_retrieval(retriever, queries, gold_doc_ids, k=5) -> Dict[str, float]:
    """P@k and MRR averaged over a list of (query, gold_set) pairs."""
    if not queries:
        return {"P@k": 0.0, "MRR": 0.0, "n": 0}
    p = 0.0; mrr = 0.0; n = 0
    for q, gold in zip(queries, gold_doc_ids):
        if not gold:
            continue
        ret = retriever.retrieve(q, top_k=k)
        ids = [r["doc_id"] for r in ret]
        p += sum(1 for d in ids if d in gold) / k
        for rank, d in enumerate(ids, 1):
            if d in gold:
                mrr += 1.0 / rank
                break
        n += 1
    return {"P@k": p / n if n else 0.0, "MRR": mrr / n if n else 0.0, "n": n}


def _verdict(train: float, test: float,
              underfit_floor: float = 0.2,
              overfit_gap: float    = 0.25) -> str:
    """Heuristic verdict for train/test metric pair."""
    if test < underfit_floor and train < underfit_floor:
        return "UNDERFIT (both low)"
    if train - test > overfit_gap:
        return f"OVERFIT (Δ={train-test:+.2f})"
    if test < train and (train - test) > 0.05:
        return f"mild overfit (Δ={train-test:+.2f})"
    if test > train:
        return "OK (test ≥ train)"
    return "OK"


# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostics(cfg_path: str = "config/full.yaml",
                     n_probe: int = 40,
                     seed: int = 42) -> Dict:
    """
    Full overfitting/underfitting/leakage diagnostic.

    n_probe : how many train / test samples to use for retrieval probes.
              40 is enough for a clear signal without blowing runtime.
    """
    sys.path.insert(0, os.getcwd())
    from utils.config import load_config
    from training.evaluation import build_system

    cfg = load_config(cfg_path)
    print("=" * 70)
    print(f"DIAGNOSTICS  (config={cfg.get('name', '?')})")
    print("=" * 70)

    with open(cfg["data"]["train_split"]) as f: train_chunks = json.load(f)
    with open(cfg["data"]["test_split"])  as f: test_chunks  = json.load(f)

    # ── Leakage check 1: document-level disjointness ────────────────────────
    train_docs = {c["doc_id"] for c in train_chunks}
    test_docs  = {c["doc_id"] for c in test_chunks}
    disjoint, overlap = _doc_disjoint(train_docs, test_docs)
    print(f"\n[Leakage: train/test docs]  "
          f"{'✓ disjoint' if disjoint else f'✗ OVERLAP ({overlap} docs)'}  "
          f"(train={len(train_docs)} docs, test={len(test_docs)} docs)")

    # ── Leakage check 2: chunk-level disjointness ───────────────────────────
    train_ids = {c["chunk_id"] for c in train_chunks}
    test_ids  = {c["chunk_id"] for c in test_chunks}
    chunk_overlap = len(train_ids & test_ids)
    print(f"[Leakage: train/test chunks]  "
          f"{'✓ disjoint' if chunk_overlap == 0 else f'✗ OVERLAP ({chunk_overlap} chunks)'}")

    # Build retriever
    print("\nBuilding retriever for retrieval-probe evaluation...")
    sysd = build_system(cfg)
    retriever = sysd["retriever"]

    rnd = random.Random(seed)

    # ── Retrieval probe: train vs test ──────────────────────────────────────
    # For train: pick n_probe chunks, make queries from first 10 words
    # (similar to how the bi-encoder was trained — so strong overfit would
    # push TRAIN metrics close to 1.0 while TEST stays low)
    tr_sample = rnd.sample(train_chunks, min(n_probe, len(train_chunks)))
    tr_queries = [" ".join(c["text"].split()[:10]) for c in tr_sample]
    tr_gold    = [{c["doc_id"]}                  for c in tr_sample]
    tr_metrics = _eval_retrieval(retriever, tr_queries, tr_gold, k=5)

    te_sample = rnd.sample(test_chunks, min(n_probe, len(test_chunks)))
    te_queries = [" ".join(c["text"].split()[:10]) for c in te_sample]
    te_gold    = [{c["doc_id"]}                  for c in te_sample]
    te_metrics = _eval_retrieval(retriever, te_queries, te_gold, k=5)

    print("\n[Retriever: overfit/underfit check]")
    print(f"  P@5      train={tr_metrics['P@k']:.3f}   test={te_metrics['P@k']:.3f}   "
          f"Δ={tr_metrics['P@k']-te_metrics['P@k']:+.3f}   "
          f"→ {_verdict(tr_metrics['P@k'], te_metrics['P@k'])}")
    print(f"  MRR      train={tr_metrics['MRR']:.3f}   test={te_metrics['MRR']:.3f}   "
          f"Δ={tr_metrics['MRR']-te_metrics['MRR']:+.3f}   "
          f"→ {_verdict(tr_metrics['MRR'], te_metrics['MRR'])}")

    # ── Fusion weights: sanity check ────────────────────────────────────────
    alpha, beta = retriever.alpha, retriever.beta
    print(f"\n[Fusion weights]")
    print(f"  α (cosine)  = {alpha:.3f}")
    print(f"  β (lexical) = {beta:.3f}")
    if abs(alpha - beta) > 0.95:
        print("  → one weight dominates; consider reducing LR or adding regularisation")
    elif abs(alpha - 0.5) < 0.05:
        print("  → near-50/50 split; fusion barely differentiated from pure mean")
    else:
        print("  → balanced — semantic and lexical both contribute")

    # ── Learned router (if present) ─────────────────────────────────────────
    router_path = "models/checkpoints/router.npz"
    if os.path.exists(router_path):
        try:
            from training.router import LearnedRouter
            r = LearnedRouter.load(router_path, encoder=retriever.encoder)

            # Training accuracy (should be high)
            from training.router import ROUTER_TRAIN
            train_probe = [(q, l) for q, l in ROUTER_TRAIN]
            correct_tr = sum(1 for q, l in train_probe if r.predict(q) == l)
            train_acc = correct_tr / len(train_probe)

            # Held-out paraphrases (test generalisation)
            held_out = [
                ("I cannot access my account at all",       "CreateTicket"),
                ("The sign-in page throws an error",        "CreateTicket"),
                ("Show me the official policy on SSI",      "GetPolicy"),
                ("How do work credits accumulate?",         "SearchKB"),
                ("Who is entitled to Medicare at 65?",      "SearchKB"),
                ("My online session keeps logging me out",  "CreateTicket"),
                ("What does regulation 4 actually say?",    "GetPolicy"),
                ("When do survivor benefits kick in?",      "SearchKB"),
            ]
            correct_te = sum(1 for q, l in held_out if r.predict(q) == l)
            test_acc = correct_te / len(held_out)

            print(f"\n[Learned router: overfit/underfit check]")
            print(f"  Accuracy  train={train_acc:.3f}   held-out={test_acc:.3f}   "
                  f"Δ={train_acc-test_acc:+.3f}   "
                  f"→ {_verdict(train_acc, test_acc, underfit_floor=0.6, overfit_gap=0.3)}")
        except Exception as e:
            print(f"\n[Learned router] could not diagnose: {e}")
    else:
        print(f"\n[Learned router] not trained (run `default_router().save(...)`)")

    # ── WEB_CACHE leakage sweep ─────────────────────────────────────────────
    from training.tools import WEB_CACHE
    from training.evaluation import HARDCODED_QUERIES
    literal_matches = [q for q, _ in HARDCODED_QUERIES
                       if q.lower().strip() in {k.lower() for k in WEB_CACHE}]
    if literal_matches:
        print(f"\n[WEB_CACHE leakage] ✗ {len(literal_matches)} eval queries match a "
              f"cache key LITERALLY — that's cheating")
        for q in literal_matches:
            print(f"    - {q!r}")
    else:
        print(f"\n[WEB_CACHE leakage] ✓ no eval query is a literal cache key")
        substr_matches = [q for q, _ in HARDCODED_QUERIES
                          if any(k in q.lower() for k in WEB_CACHE)]
        if substr_matches:
            print(f"  ({len(substr_matches)} queries share keywords with cache — OK, "
                  f"cache is just a response-time optimisation, not a source of truth)")

    print("\n" + "=" * 70)
    print("Diagnostic complete.")
    print("=" * 70)

    return {
        "leakage_docs_disjoint":   disjoint,
        "leakage_chunks_overlap":  chunk_overlap,
        "retriever_train_P@5":     tr_metrics["P@k"],
        "retriever_test_P@5":      te_metrics["P@k"],
        "retriever_train_MRR":     tr_metrics["MRR"],
        "retriever_test_MRR":      te_metrics["MRR"],
        "fusion_alpha":            alpha,
        "fusion_beta":             beta,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/full.yaml")
    args = ap.parse_args()
    run_diagnostics(args.config)
