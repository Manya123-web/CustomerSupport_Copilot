"""Metric sanity tests — no model downloads, runs in <1 second."""
from utils.metrics import (precision_at_k, recall_at_k, mrr,
                             grounding_score, escalation_rate,
                             correct_escalation_rate)


def test_precision_recall_mrr():
    ret = ["a", "b", "c", "d", "e"]
    rel = {"c", "e"}
    assert precision_at_k(ret, rel, 5) == 0.4
    assert recall_at_k(ret, rel, 5) == 1.0
    assert mrr(ret, rel) == 1 / 3          # 'c' is rank 3
    assert precision_at_k(ret, rel, 0) == 0.0


def test_grounding_bounds():
    ctx = "Social Security benefits require 40 work credits."
    assert grounding_score("Social Security needs 40 credits.", ctx) > 0.5
    assert grounding_score("", ctx) == 0.0
    assert 0.0 <= grounding_score("completely unrelated text here", ctx) <= 1.0


def test_escalation_and_cer():
    outs = [{"ticket": {"id": "X"}}, {"ticket": None}, {"ticket": None}]
    assert abs(escalation_rate(outs) - 1/3) < 1e-9
    # all kb-sufficient → CER = 1 - escalation_rate
    assert abs(correct_escalation_rate(outs) - 2/3) < 1e-9
    # mixed kb_sufficient
    assert correct_escalation_rate(outs, [False, True, True]) == 1.0
