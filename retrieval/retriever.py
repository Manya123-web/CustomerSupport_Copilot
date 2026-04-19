"""
retrieval/retriever.py
----------------------
Hybrid retriever: α · cosine + β · Jaccard-lexical, with α+β=1 enforced
by a softmax parametrisation and learned via manual-gradient BPR training
(heavy-ball momentum, gradient clipping).

Why manual gradients?
    The fusion layer has exactly two parameters. Writing the chain rule by
    hand keeps the training loop fully transparent and removes PyTorch as a
    dependency for this component — which is one of the novelty claims.

Why softmax instead of clipping α, β ∈ [0, 1]?
    Softmax guarantees α+β=1 AND both positive at every step, so the update
    rule never has to project back onto the simplex.
"""
from __future__ import annotations
import math
import random
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

from utils.metrics import FULL_STOP   # single source of truth for stopwords


# ── Lexical scorer (Jaccard over non-stop tokens) ────────────────────────────
_W = re.compile(r"\w+")

def lexical_overlap(q: str, d: str) -> float:
    qw = set(_W.findall(q.lower())) - FULL_STOP
    dw = set(_W.findall(d.lower())) - FULL_STOP
    if not qw or not dw:
        return 0.0
    return len(qw & dw) / len(qw | dw)


# ── Softmax / sigmoid / clip helpers ─────────────────────────────────────────
def _softmax2(la: float, lb: float) -> Tuple[float, float]:
    m = max(la, lb)
    ea, eb = math.exp(la - m), math.exp(lb - m)
    s = ea + eb
    return ea / s, eb / s

def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def _clip(g: float, lim: float) -> float:
    return max(-lim, min(lim, g))


# ── Trainable fusion module ──────────────────────────────────────────────────
class LearnableFusion:
    """
    Two-parameter (logit) fusion module trained with pairwise BPR.
    Methods:
        fit(triplets, ...)   — run manual-gradient descent
        alpha_beta()         — current fusion weights
    """
    def __init__(self, lr: float = 0.5, momentum: float = 0.9,
                 grad_clip: float = 5.0, n_epochs: int = 80):
        self.lr          = lr
        self.momentum    = momentum
        self.grad_clip   = grad_clip
        self.n_epochs    = n_epochs
        self.l_alpha     = 0.0
        self.l_beta      = 0.0
        self.v_alpha     = 0.0
        self.v_beta      = 0.0
        self.history: List[Dict[str, float]] = []

    def alpha_beta(self) -> Tuple[float, float]:
        return _softmax2(self.l_alpha, self.l_beta)

    def fit(self, triplets: List[Tuple[str, int, int]],
            chunk_embs: np.ndarray,
            q_emb_matrix: np.ndarray, q_to_row: Dict[str, int],
            lex_cache: Dict[Tuple[str, int, int], Tuple[float, float]]):
        """
        triplets      : list of (query_str, pos_idx, neg_idx)
        chunk_embs    : shape (N, D), unit-normalised
        q_emb_matrix  : shape (Q, D), unit-normalised, pre-encoded ONCE
        lex_cache     : {(q, pos, neg): (lex_pos, lex_neg)}

        Note: `chunk_texts` was removed in v2.6 — the parameter was
        received but never read (auditor #7).
        """
        for epoch in range(self.n_epochs):
            total_loss = g_a_sum = g_b_sum = 0.0
            random.shuffle(triplets)
            for (q, pi, ni) in triplets:
                a, b = _softmax2(self.l_alpha, self.l_beta)
                qv   = q_emb_matrix[q_to_row[q]]
                cp   = float(np.dot(qv, chunk_embs[pi].astype("float64")))
                cn   = float(np.dot(qv, chunk_embs[ni].astype("float64")))
                lp, ln = lex_cache[(q, pi, ni)]
                margin = a * (cp - cn) + b * (lp - ln)
                p = _sigmoid(margin)
                total_loss += -math.log(p + 1e-10)

                # Gradients (chain rule, written out)
                dL_dm  = p - 1.0
                dL_da  = dL_dm * (cp - cn)
                dL_db  = dL_dm * (lp - ln)
                g_a_sum += dL_da * a * (1 - a) + dL_db * (-b * a)
                g_b_sum += dL_db * b * (1 - b) + dL_da * (-a * b)

            n = len(triplets)
            g_a = _clip(g_a_sum / n, self.grad_clip)
            g_b = _clip(g_b_sum / n, self.grad_clip)
            # heavy-ball momentum
            self.v_alpha = self.momentum * self.v_alpha - self.lr * g_a
            self.v_beta  = self.momentum * self.v_beta  - self.lr * g_b
            self.l_alpha += self.v_alpha
            self.l_beta  += self.v_beta
            a_now, b_now = _softmax2(self.l_alpha, self.l_beta)
            self.history.append({"epoch": epoch + 1,
                                 "loss": total_loss / n,
                                 "alpha": a_now, "beta": b_now})
        return self


def build_triplets(train_chunks: List[Dict[str, Any]],
                   n_triplets: int = 300, seed: int = 42):
    rnd = random.Random(seed)
    d2i = defaultdict(list)
    for i, c in enumerate(train_chunks):
        d2i[c["doc_id"]].append(i)
    doc_ids = list(d2i.keys())
    # BUG FIX (auditor #3): was doc_ids[:n_triplets] — first-n bias.
    # Sample uniformly at random with a fixed seed. If n_triplets > total
    # available docs, oversample with replacement.
    if n_triplets <= len(doc_ids):
        sampled_docs = rnd.sample(doc_ids, n_triplets)
    else:
        sampled_docs = [rnd.choice(doc_ids) for _ in range(n_triplets)]
    trips = []
    for did in sampled_docs:
        pi = rnd.choice(d2i[did])
        nd = rnd.choice([d for d in doc_ids if d != did])
        ni = rnd.choice(d2i[nd])
        q  = " ".join(train_chunks[pi]["text"].split()[:10])
        trips.append((q, pi, ni))
    return trips, d2i


# ── FAISS index ──────────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray, path: str | None = None) -> faiss.Index:
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings.astype("float32"))
    if path:
        faiss.write_index(idx, path)
    return idx


# ── Retrieve helper (used by the agent loop) ─────────────────────────────────
class HybridRetriever:
    """Wraps a FAISS index + encoder + fusion weights into one `retrieve()`."""

    def __init__(self, faiss_index: faiss.Index, encoder,
                 chunks: List[Dict[str, Any]],
                 alpha: float = 1.0, beta: float = 0.0):
        self.index   = faiss_index
        self.encoder = encoder
        self.chunks  = chunks
        self.alpha   = alpha
        self.beta    = beta

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = self.encoder.encode([query], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype("float32")
        D, I = self.index.search(qv[None, :], top_k * 4)
        out = []
        for score, idx in zip(D[0], I[0]):
            c = self.chunks[idx]
            lex = lexical_overlap(query, c["text"])
            out.append({**c,
                        "cos_score": float(score),
                        "lex_score": lex,
                        "score": self.alpha * float(score) + self.beta * lex})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]
