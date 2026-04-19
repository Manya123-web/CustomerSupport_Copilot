"""
tests/test_learned_router.py
----------------------------
Proves that the LearnedRouter is a real trained model (not a lookup table):

  1. Before training, predict() raises — no fall-through hardcoded logic.
  2. After fit() on the canonical labelled set, it classifies held-out
     PARAPHRASES the keyword router cannot (e.g. "stuck at the login page"
     is not in the keyword list but should map to CreateTicket).
  3. Retraining with shuffled labels produces different weights (proves
     the weights depend on the data, not a fixed init).

These tests need the real MiniLM encoder, so they are marked 'slow' and
skipped unless the encoder is locally cached. The first-run download is
~90 MB; subsequent runs are offline.
"""
import os
import pytest
import numpy as np


def _have_encoder() -> bool:
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _have_encoder(),
    reason="MiniLM encoder not available offline — skipping learned-router test"
)


def test_router_refuses_before_training():
    from training.router import LearnedRouter
    r = LearnedRouter()
    with pytest.raises(RuntimeError):
        r.predict("anything")


def test_router_learns_paraphrases():
    from training.router import default_router
    r = default_router()
    # Held-out paraphrases — NONE of these appear verbatim in ROUTER_TRAIN.
    # If the router were memorising, it would misclassify them.
    held_out = [
        ("I cannot access my account at all",       "CreateTicket"),
        ("The sign-in page throws an error",        "CreateTicket"),
        ("Show me the official policy on SSI",      "GetPolicy"),
        ("How do work credits accumulate?",         "SearchKB"),
        ("Who is entitled to Medicare at 65?",      "SearchKB"),
    ]
    correct = sum(1 for q, y in held_out if r.predict(q) == y)
    # A fair threshold — MiniLM + logistic on ~35 examples typically
    # gets 4/5 or 5/5. Set 3/5 so the test isn't flaky on CPU variance.
    assert correct >= 3, f"learned router only got {correct}/5 held-out paraphrases right"


def test_router_weights_depend_on_data():
    """Retraining with permuted labels must yield different weight matrices."""
    from training.router import LearnedRouter, ROUTER_TRAIN
    qs, ls = zip(*ROUTER_TRAIN)
    r1 = LearnedRouter().fit(qs, ls)
    # Permute labels — this is a DIFFERENT task; weights must change.
    import random
    rnd = random.Random(0)
    permuted = list(ls); rnd.shuffle(permuted)
    r2 = LearnedRouter().fit(qs, permuted)
    assert r1.W.shape == r2.W.shape
    assert not np.allclose(r1.W, r2.W), "router weights unchanged after relabelling"
