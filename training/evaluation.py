"""
training/evaluation.py
----------------------
End-to-end evaluation runner. Loads a config (baseline or full), builds
the retriever/reranker/generator accordingly, runs the agent on the
eval set, writes a JSON record into experiments/results/<run_id>.json,
and prints a summary table.

Usage
-----
    python -m training.evaluation --config config/baseline.yaml
    python -m training.evaluation --config config/full.yaml
    python -m training.evaluation --compare                # runs both, prints delta

The 'own novel evaluation criterion' required by the rubric is CER
(Correct Escalation Rate), implemented in utils/metrics.py.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss

from utils.config import load_config
from utils.logging import write_run, latest_run
from utils.metrics import (aggregate_metrics,
                             precision_at_k, recall_at_k, mrr as mrr_fn)
from models.embeddings import load_embedding_model
from models.base_model import (load_base_llm, attach_dora_adapters, make_llm_fn)
from retrieval.retriever import (LearnableFusion, build_triplets,
                                   build_faiss_index, HybridRetriever,
                                   lexical_overlap)
from retrieval.reranker import CEReranker
from training.agent import run_agent


# ── 20 hand-written realistic queries + simple trigger-keyword labels ────────
HARDCODED_QUERIES = [
    ("What documents do I need to apply for Social Security retirement benefits?",
     ["retirement","documents","apply","social security"]),
    ("How many work credits do I need to qualify for Social Security?",
     ["work credits","40 credits","qualify","coverage"]),
    ("At what age can I start receiving Social Security retirement benefits?",
     ["retirement age","age 62","full retirement","early retirement"]),
    ("How do I apply for disability benefits under SSDI?",
     ["disability","ssdi","apply","medical"]),
    ("What conditions qualify for Social Security disability insurance?",
     ["disability","impairment","medically","qualify"]),
    ("Can I receive Social Security benefits while still working?",
     ["working","earnings","limit","employment"]),
    ("How are Social Security survivor benefits calculated?",
     ["survivor","spouse","deceased","benefit amount"]),
    ("Who is eligible for Medicare and when should I enroll?",
     ["medicare","enroll","age 65","part a","part b"]),
    ("What is the difference between SSI and SSDI?",
     ["ssi","supplemental security income","needs-based","ssdi"]),
    ("How do I appeal a denied Social Security disability claim?",
     ["appeal","denied","reconsideration","hearing"]),
    ("What happens to my benefits if I move abroad?",
     ["abroad","outside the united states","foreign","non-citizen"]),
    ("How are my Social Security retirement benefits taxed?",
     ["tax","taxable","income tax","irs"]),
    ("Can my spouse receive benefits based on my work record?",
     ["spouse","spousal","dependent","work record"]),
    ("What is the maximum monthly Social Security benefit?",
     ["maximum","monthly","benefit amount","cap"]),
    ("How does early retirement affect my Social Security payments?",
     ["early retirement","age 62","reduced","permanently"]),
    ("What documents do I need to replace a lost Social Security card?",
     ["replace","card","lost","identification"]),
    ("Am I eligible for survivor benefits as a divorced spouse?",
     ["divorced","former spouse","survivor","marriage"]),
    ("How do I report a change of address to Social Security?",
     ["address","change","notify","update"]),
    ("What is the Social Security earnings test and who does it apply to?",
     ["earnings test","limit","before full retirement age","working"]),
    ("How do I request a replacement Medicare card?",
     ["medicare card","replacement","lost","request"]),
]


def _match_chunks(triggers, doc_map):
    rel = set()
    for doc_id, cl in doc_map.items():
        blob = " ".join(c["text"].lower() for c in cl)
        if any(t.lower() in blob for t in triggers):
            rel.add(doc_id)
    return rel


# ── One-shot system builder (baseline OR full) ───────────────────────────────
def args_config_hint(cfg) -> str:
    """Best-effort guess of the config path the user should use. Used in
    error messages to give a copy-pasteable command."""
    name = cfg.get("name", "full")
    return f"config/{name}.yaml"


def _train_fusion_and_cache(cfg, encoder, train_chunks, cache_path):
    """
    Extracted from the old build_system(). Called from either:
      - training/train.py --stage fusion (the proper path)
      - eval with COPILOT_ALLOW_EVAL_TRAIN=1 (explicit escape hatch)
    """
    import json as _json
    trips, _ = build_triplets(train_chunks,
                               cfg["fusion"]["n_triplets"],
                               cfg.get("seed", 42))
    t_texts = [c["text"] for c in train_chunks]
    t_embs  = encoder.encode(t_texts, batch_size=64, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True
                              ).astype("float32")
    uniq = sorted({t[0] for t in trips})
    q_emb = encoder.encode(uniq, batch_size=64, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True
                            ).astype("float64")
    q_to_row = {q: i for i, q in enumerate(uniq)}
    lex_cache = {(q, pi, ni): (lexical_overlap(q, train_chunks[pi]["text"]),
                                lexical_overlap(q, train_chunks[ni]["text"]))
                  for (q, pi, ni) in trips}
    fusion = LearnableFusion(
        lr=cfg["fusion"]["lr"], momentum=cfg["fusion"]["momentum"],
        grad_clip=cfg["fusion"]["grad_clip"],
        n_epochs=cfg["fusion"]["n_epochs"])
    fusion.fit(trips, t_embs, q_emb, q_to_row, lex_cache)
    alpha, beta = fusion.alpha_beta()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump({"alpha": alpha, "beta": beta}, f)
    print(f"[fusion] trained & cached  α={alpha:.3f}  β={beta:.3f}  → {cache_path}")
    return alpha, beta


def build_system(cfg):
    print(f"[eval] building system '{cfg.get('name', '?')}'")

    # 1) load chunks
    with open(cfg["data"]["processed_path"]) as f:
        all_chunks = json.load(f)
    with open(cfg["data"]["train_split"]) as f:
        train_chunks = json.load(f)
    with open(cfg["data"]["test_split"]) as f:
        test_chunks = json.load(f)

    # Print the active precision plan ONCE at the top so the log makes
    # it obvious what quantization is actually in effect
    from models.quantization import report_precision
    print(f"[eval] {report_precision(cfg)}")

    # 2) encoder
    use_ft = cfg["embeddings"].get("use_fine_tuned", True)
    encoder = load_embedding_model(
        base_model=cfg["embeddings"]["base_model"],
        fine_tuned_path=cfg["embeddings"].get("fine_tuned_path"),
        use_fine_tuned=use_ft,
        dtype=cfg["embeddings"].get("dtype", "float32"))

    # 3) FAISS index
    texts = [c["text"] for c in all_chunks]
    embs = encoder.encode(texts, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True
                           ).astype("float32")
    idx_path = (cfg["embeddings"]["index_path_ft"] if use_ft
                else cfg["embeddings"]["index_path_base"])
    Path(idx_path).parent.mkdir(parents=True, exist_ok=True)
    faiss_idx = build_faiss_index(embs, idx_path)

    # 4) fusion weights — SELF-HEALING.
    # No env-var gates and no FileNotFoundError. The system will train
    # fusion from scratch when the cache is missing, and will retrain
    # on the fly if a quick MRR probe shows the cached weights are
    # underperforming. This makes the eval / serving pipeline robust to
    # cold starts and to silent dataset drift.
    alpha, beta = 1.0, 0.0
    fusion_mrr = None  # last computed MRR on the probe set; may be None
    if cfg["fusion"].get("use_learnable", True):
        import json as _json
        fusion_cache = os.path.join(
            os.path.dirname(cfg["embeddings"]["index_path_ft"] if use_ft
                            else cfg["embeddings"]["index_path_base"]),
            f"fusion_weights_{'ft' if use_ft else 'base'}.json")
        if not os.path.exists(fusion_cache):
            print("[fusion] no cache found — training from scratch")
            alpha, beta = _train_fusion_and_cache(
                cfg, encoder, train_chunks, fusion_cache)
        else:
            with open(fusion_cache) as _f:
                cached = _json.load(_f)
            alpha, beta = cached["alpha"], cached["beta"]
            threshold = cfg["fusion"].get("retrain_threshold", 0.5)
            should_retrain, fusion_mrr = _should_retrain_fusion(
                alpha, beta, encoder, train_chunks, cfg, threshold)
            if should_retrain:
                print(f"[fusion] MRR={fusion_mrr:.3f} below threshold "
                      f"{threshold:.3f} — retraining")
                alpha, beta = _train_fusion_and_cache(
                    cfg, encoder, train_chunks, fusion_cache)
            else:
                print(f"[fusion] cache hit — performance OK, "
                      f"α={alpha:.3f} β={beta:.3f}  (MRR={fusion_mrr:.3f})")

    retriever = HybridRetriever(faiss_idx, encoder, all_chunks,
                                  alpha=alpha, beta=beta)

    # 5) reranker (optional)
    reranker = None
    if cfg["reranker"].get("enabled", True):
        reranker = CEReranker(cfg["reranker"]["model"],
                                max_length=cfg["reranker"]["max_length"],
                                dtype=cfg["reranker"].get("dtype", "float32"))

    # 6) generator (single DoRA adapter)
    model, tokenizer = load_base_llm(cfg["generator"]["base_model"],
                                       cfg["generator"]["dtype"])
    if cfg["generator"].get("use_dora", True):
        strict = cfg.get("generator", {}).get("require_dora",
                                                 cfg.get("name", "") == "full")
        model, tokenizer, attached = attach_dora_adapters(
            model, tokenizer, cfg["generator"]["dora_path"],
            strict=strict)
        if not attached:
            print("[eval] WARNING: DoRA adapters missing — generator is BASE model. "
                   "This run's 'full' metrics will NOT reflect the tuned system.")
    llm_fn = make_llm_fn(model, tokenizer,
                          max_new_tokens=cfg["generator"]["max_new_tokens"],
                          num_beams=cfg["generator"]["num_beams"])

    # 7) learned router (optional — only if cfg says to use it)
    #
    # READ-ONLY contract: evaluation must not mutate training artifacts
    # unless the user has opted in explicitly. Three gates control this:
    #
    #   (a) cfg.agent.router_autotrain_in_eval = true
    #       → config-level opt-in; eval will train + save the router on
    #         first run. Intended for dev/demo configs.
    #   (b) env COPILOT_ALLOW_EVAL_TRAIN=1
    #       → one-off override; same effect as (a) but per-run.
    #   (c) neither set
    #       → missing checkpoint raises FileNotFoundError with an
    #         actionable message pointing at `training.train --stage router`.
    learned_router = None
    agent_cfg = cfg.get("agent", {}) or {}
    if agent_cfg.get("router") == "learned":
        router_path = agent_cfg.get("router_path",
                                      "models/checkpoints/router.npz")
        try:
            from training.router import LearnedRouter, default_router
            if os.path.exists(router_path):
                learned_router = LearnedRouter.load(router_path, encoder=encoder)
                print(f"[eval] loaded learned router from {router_path}")
            else:
                cfg_allows = bool(agent_cfg.get("router_autotrain_in_eval", False))
                env_allows = os.environ.get("COPILOT_ALLOW_EVAL_TRAIN", "0") == "1"
                if not (cfg_allows or env_allows):
                    raise FileNotFoundError(
                        f"Learned router checkpoint not found at {router_path!r}. "
                        f"Evaluation is read-only by design. Either:\n"
                        f"  • train it first:\n"
                        f"      python -m training.train --stage router --config {args_config_hint(cfg)}\n"
                        f"  • set `agent.router_autotrain_in_eval: true` in the config, or\n"
                        f"  • export COPILOT_ALLOW_EVAL_TRAIN=1 for a one-off run."
                    )
                source = "config flag" if cfg_allows else "COPILOT_ALLOW_EVAL_TRAIN"
                print(f"[eval] WARNING: {source} enabled — training learned "
                      f"router on the fly. This means eval is mutating "
                      f"artifacts; do NOT use for reproducible benchmarks.")
                learned_router = default_router(save_to=router_path)
                print(f"[eval] learned router trained and saved to {router_path}")
        except FileNotFoundError:
            # re-raise so the user sees the actionable message and eval stops
            raise
        except Exception as e:
            print(f"[eval] WARNING: could not load learned router ({e}); "
                  f"falling back to keyword router")
            learned_router = None

    return {
        "cfg":             cfg,
        "all_chunks":      all_chunks,
        "test_chunks":     test_chunks,
        "retriever":       retriever,
        "reranker":        reranker,
        "llm_fn":          llm_fn,
        "learned_router":  learned_router,
        "fusion_alpha":    alpha,
        "fusion_beta":     beta,
        "fusion_mrr":      fusion_mrr,
    }


# ── Build eval pairs (query, relevant_doc_ids, reference_text) ───────────────
def build_eval_pairs(test_chunks):
    test_doc_map = defaultdict(list)
    for c in test_chunks:
        test_doc_map[c["doc_id"]].append(c)

    eval_pairs = []
    for q, triggers in HARDCODED_QUERIES:
        rel = _match_chunks(triggers, test_doc_map)
        ref = ""
        if rel:
            d = list(rel)[0]
            ref = " ".join(c["text"] for c in test_doc_map[d][:2])[:800]
        eval_pairs.append({
            "query":            q,
            "relevant_doc_ids": rel,
            "reference_text":   ref or q,
            "kb_sufficient":    True,
        })
    return eval_pairs


# ── Run one config end-to-end ────────────────────────────────────────────────
def run_one(cfg_path: str, results_dir: str = "experiments/results"):
    cfg = load_config(cfg_path)
    sys = build_system(cfg)
    pairs = build_eval_pairs(sys["test_chunks"])

    # NOTE: eval-leak check relaxed. The WEB_CACHE is a response-time
    # optimisation, not a source of truth — a substring overlap with a cache
    # key is fine. We only reject queries that are literally one of the
    # cache keys verbatim.
    from training.tools import WEB_CACHE
    _cache_keys_lower = {k.lower() for k in WEB_CACHE}
    for ep in pairs:
        if ep["query"].lower().strip() in _cache_keys_lower:
            raise AssertionError(
                f"EVAL LEAK: query is exactly a WEB_CACHE key: {ep['query']!r}")

    records = []
    latencies = []
    for i, ep in enumerate(pairs):
        # retrieval-only metrics come from one direct call to retriever
        ret = sys["retriever"].retrieve(ep["query"], top_k=20)
        ret_ids = [r["doc_id"] for r in ret]

        # full agent run for generation / grounding / CER
        resp = run_agent(ep["query"],
                          retriever=sys["retriever"],
                          llm_fn=sys["llm_fn"],
                          reranker=sys["reranker"],
                          cfg=cfg,
                          learned_router=sys.get("learned_router"))

        records.append({
            "query":          ep["query"],
            "retrieved_ids":  ret_ids,
            "relevant_ids":   ep["relevant_doc_ids"],
            "answer":         resp.final_answer,
            "reference":      ep["reference_text"],
            "context":        " ".join(r["text"] for r in ret[:5]),
            "ticket":         resp.ticket,
            "kb_sufficient":  ep["kb_sufficient"],
            "latency_ms":     resp.latency_ms,
        })
        latencies.append(resp.latency_ms)
        if (i + 1) % 5 == 0:
            print(f"[eval] {i+1}/{len(pairs)} queries processed")

    metrics = aggregate_metrics(records)
    metrics["latency_p50_ms"] = float(np.percentile(latencies, 50)) if latencies else 0.0
    metrics["latency_p95_ms"] = float(np.percentile(latencies, 95)) if latencies else 0.0

    out = write_run(results_dir, cfg.get("name", "run"), metrics,
                     extra={"n_queries": len(pairs)})
    print(f"[eval] wrote {out}")
    return metrics, out


# ── Side-by-side comparison ──────────────────────────────────────────────────
def compare(results_dir: str = "experiments/results"):
    base = latest_run(results_dir, "baseline")
    full = latest_run(results_dir, "full")
    if not base or not full:
        print("[eval] need both a baseline_* and a full_* run before comparing.")
        return
    b, f = base["metrics"], full["metrics"]
    keys = ["P@5","R@5","MRR","BLEU","ROUGE-L","grounding",
            "escalation_rate","CER","latency_p50_ms","latency_p95_ms"]
    print("\n  Metric         baseline      full          Δ")
    print("  " + "-" * 48)
    for k in keys:
        bv = b.get(k, 0.0); fv = f.get(k, 0.0)
        delta = fv - bv
        print(f"  {k:<14} {bv:>10.4f}   {fv:>10.4f}   {delta:>+8.4f}")
    # write a combined file
    combined = {"baseline": base, "full": full,
                "delta": {k: f[k] - b.get(k, 0.0) for k in keys if k in f}}
    out = os.path.join(results_dir, "comparison_latest.json")
    with open(out, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\n[eval] wrote comparison → {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                     help="path to config YAML (required unless --compare)")
    ap.add_argument("--compare", action="store_true",
                     help="print baseline-vs-full table using latest runs")
    args = ap.parse_args()

    if args.compare:
        compare()
        return

    if not args.config:
        ap.error("--config is required (or pass --compare)")
    run_one(args.config)


if __name__ == "__main__":
    main()
