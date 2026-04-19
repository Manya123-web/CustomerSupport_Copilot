"""tests/test_demo_questions.py — fast offline check."""
from training.demo_questions import (
    all_demo_questions, demo_by_category,
    STANDARD, PARAPHRASED, ACCOUNT_ISSUE, OUT_OF_SCOPE,
)


def test_counts():
    # 8 + 8 + 7 + 7 = 30
    assert len(STANDARD)      == 8
    assert len(PARAPHRASED)   == 8
    assert len(ACCOUNT_ISSUE) == 7
    assert len(OUT_OF_SCOPE)  == 7
    assert sum(1 for _ in all_demo_questions()) == 30


def test_shape():
    for item in all_demo_questions():
        assert set(item.keys()) == {"category", "query", "triggers"}
        assert isinstance(item["query"], str)
        assert len(item["query"]) > 5


def test_categories_unique():
    cats = {q for q in demo_by_category().keys()}
    assert cats == {"STANDARD", "PARAPHRASED", "ACCOUNT_ISSUE", "OUT_OF_SCOPE"}


def test_account_issue_have_account_keywords():
    """Every account-issue query should contain a ticket-routable keyword,
    so the keyword router doesn't misroute it to SearchKB."""
    TICKET_KW = ("log", "password", "sign in", "lock", "access denied",
                  "broken", "error", "account")
    for q, _ in ACCOUNT_ISSUE:
        assert any(k in q.lower() for k in TICKET_KW), \
            f"account-issue query lacks any router keyword: {q!r}"
