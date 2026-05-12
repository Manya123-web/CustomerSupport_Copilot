"""
training/router.py
------------------
Learned router for the tool-selection step.

The default router in agent.py is keyword-based: deterministic, fast, and
easy to debug — but it fails on paraphrases (e.g. "stuck at the login
screen" should go to EscalateIssue but doesn't match the keyword list).

This module implements a TRAINABLE alternative. It fits a multinomial
logistic-regression head on top of frozen sentence-transformer embeddings.
No hardcoded rules; it learns the classes {KBLookup, PolicyFetch,
EscalateIssue} from labelled examples.

Usage
-----
    from training.router import LearnedRouter
    r = LearnedRouter()
    r.fit(queries, labels)      # labels: list of "KBLookup" | "PolicyFetch" | "EscalateIssue"
    r.predict("I can't log in") # -> "EscalateIssue"
    r.save("models/checkpoints/router.npz")
    r2 = LearnedRouter.load("models/checkpoints/router.npz", encoder=r.encoder)

Switch it on via config:
    agent:
      router: "learned"         # instead of "keyword"
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np


class LearnedRouter:
    """Frozen MiniLM + logistic-regression head trained on labelled intents."""

    CLASSES = ("KBLookup", "PolicyFetch", "EscalateIssue")

    def __init__(self, encoder=None,
                 base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dtype: str = "float16"):
        if encoder is None:
            # Route through the central quantization helper so precision
            # is consistent with the retriever's encoder.
            from sentence_transformers import SentenceTransformer
            from models.quantization import quantize_sentence_transformer
            encoder = SentenceTransformer(base_model)
            encoder = quantize_sentence_transformer(encoder, dtype)
        self.encoder = encoder
        # Weights set by fit() or load()
        self.W: np.ndarray | None = None   # shape (D, C)
        self.b: np.ndarray | None = None   # shape (C,)

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(self, queries: Sequence[str], labels: Sequence[str],
            epochs: int = 200, lr: float = 0.5,
            l2: float = 1e-3, seed: int = 42) -> "LearnedRouter":
        """
        Multinomial logistic regression trained by batch gradient descent
        on frozen 384-dim embeddings. No PyTorch required — this runs
        entirely on numpy and trains in <1 s on 50 examples.
        """
        assert len(queries) == len(labels)
        X = self.encoder.encode(list(queries), convert_to_numpy=True,
                                  normalize_embeddings=True).astype("float64")
        y = np.array([self.CLASSES.index(l) for l in labels])
        N, D = X.shape
        C    = len(self.CLASSES)
        rng = np.random.default_rng(seed)
        W = rng.normal(scale=0.01, size=(D, C))
        b = np.zeros(C)

        Y = np.zeros((N, C))                 # one-hot
        Y[np.arange(N), y] = 1.0

        for ep in range(epochs):
            logits = X @ W + b              # (N, C)
            logits -= logits.max(axis=1, keepdims=True)   # numerical safety
            exp    = np.exp(logits)
            probs  = exp / exp.sum(axis=1, keepdims=True)
            # cross-entropy gradient = probs - onehot
            grad_logits = (probs - Y) / N
            grad_W = X.T @ grad_logits + l2 * W
            grad_b = grad_logits.sum(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b

        self.W, self.b = W, b
        return self

    # ── Inference ────────────────────────────────────────────────────────────
    def predict(self, query: str) -> str:
        if self.W is None:
            raise RuntimeError("LearnedRouter not fitted; call fit() or load() first")
        x      = self.encoder.encode([query], convert_to_numpy=True,
                                      normalize_embeddings=True)[0].astype("float64")
        logits = x @ self.W + self.b
        return self.CLASSES[int(np.argmax(logits))]

    def predict_proba(self, query: str) -> dict:
        x      = self.encoder.encode([query], convert_to_numpy=True,
                                      normalize_embeddings=True)[0].astype("float64")
        logits = x @ self.W + self.b
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        return {c: float(p[i]) for i, c in enumerate(self.CLASSES)}

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str):
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b,
                  classes=np.array(self.CLASSES))

    @classmethod
    def load(cls, path: str, encoder=None) -> "LearnedRouter":
        npz = np.load(path, allow_pickle=True)
        r = cls(encoder=encoder)
        r.W, r.b = npz["W"], npz["b"]
        return r


# ── Canonical labelled training set for the router ──────────────────────────
# ~40 examples spanning paraphrases of each intent. Small on purpose — this
# is an intent-classification problem on a 3-way label space; with frozen
# sentence-transformer embeddings, a logistic head reaches >90% val accuracy
# on this size.
ROUTER_TRAIN = [
    # KBLookup
    ("What documents do I need to apply for benefits?",             "KBLookup"),
    ("How many work credits are needed for retirement?",             "KBLookup"),
    ("Who qualifies for disability insurance?",                      "KBLookup"),
    ("At what age can I start collecting retirement benefits?",      "KBLookup"),
    ("Explain the difference between SSI and SSDI",                  "KBLookup"),
    ("What is the earnings test?",                                   "KBLookup"),
    ("How do I appeal a denied claim?",                              "KBLookup"),
    ("Are Social Security benefits taxable?",                        "KBLookup"),
    ("Can a spouse claim benefits on my record?",                    "KBLookup"),
    ("How are survivor benefits calculated?",                        "KBLookup"),
    ("When should I enroll in Medicare?",                            "KBLookup"),
    ("What is the maximum monthly benefit?",                         "KBLookup"),
    ("How do I request a new Social Security card?",                 "KBLookup"),
    ("How long does processing take?",                               "KBLookup"),
    # PolicyFetch
    ("What does section 1 say?",                                     "PolicyFetch"),
    ("Show me the policy on disability",                             "PolicyFetch"),
    ("Which regulation governs survivor benefits?",                  "PolicyFetch"),
    ("Give me the rule for Medicare enrollment",                     "PolicyFetch"),
    ("Cite the statute for earnings test",                           "PolicyFetch"),
    ("Pull up section 3",                                            "PolicyFetch"),
    ("What is the policy for SSI?",                                  "PolicyFetch"),
    ("Rule book on early retirement",                                "PolicyFetch"),
    # EscalateIssue
    ("I can't log into my account",                                  "EscalateIssue"),
    ("Stuck at the login page",                                      "EscalateIssue"),
    ("Forgot my password, help",                                     "EscalateIssue"),
    ("My account is locked",                                         "EscalateIssue"),
    ("I am locked out of the portal",                                "EscalateIssue"),
    ("Sign in is broken",                                            "EscalateIssue"),
    ("Getting an error when I try to access my benefits",            "EscalateIssue"),
    ("Unable to reset my password",                                  "EscalateIssue"),
    ("My account shows the wrong information",                       "EscalateIssue"),
    ("I keep getting access denied",                                 "EscalateIssue"),
    ("Website not working for me",                                   "EscalateIssue"),
    ("My portal says I'm locked out",                                "EscalateIssue"),
]


def default_router(save_to: str | None = None) -> LearnedRouter:
    """Train a router on the canonical labelled set and optionally save it."""
    queries, labels = zip(*ROUTER_TRAIN)
    r = LearnedRouter().fit(queries, labels)
    if save_to:
        r.save(save_to)
    return r
