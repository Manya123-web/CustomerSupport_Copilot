"""
training/test_questions_v2.py
-----------------------------
30 diverse evaluation questions covering the six categories the
auditor asked for:

  1. Simple queries (5)
  2. Complex reasoning queries (5)
  3. Multi-turn / context-dependent queries (5 two-turn pairs = 10 turns)
  4. Edge cases / ambiguous queries (5)
  5. Policy-related queries (5)  — tests GetPolicy's topic-mapper
  6. Memory-based queries (embedded in #3)

Each entry has:
    category      : which bucket this belongs to
    query         : the user utterance
    triggers      : keywords expected in the answer (None for open-ended)
    expected_tool : what the router should pick
    expected_grd  : True iff we expect a grounded answer (not a ticket)
    followup_of   : index of the previous turn this depends on (if any)

Use all_test_questions() to iterate in presentation order.
"""

# ── Category 1: Simple queries (single-hop KB lookup) ────────────────────────
SIMPLE = [
    dict(category="simple",
         query="What is the full retirement age for Social Security?",
         triggers=["retirement age", "67"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="simple",
         query="How many work credits do I need to qualify for retirement?",
         triggers=["credits", "40"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="simple",
         query="What is SSI?",
         triggers=["supplemental", "income"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="simple",
         query="When does Medicare enrollment start?",
         triggers=["medicare", "65"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="simple",
         query="At what age can a surviving spouse claim benefits?",
         triggers=["spouse", "60"],
         expected_tool="SearchKB", expected_grd=True),
]

# ── Category 2: Complex reasoning (multi-fact synthesis) ─────────────────────
COMPLEX = [
    dict(category="complex",
         query="If I retire at 63 with 38 work credits, what happens to my benefits?",
         triggers=["credits", "early", "reduced"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="complex",
         query="Can I collect both SSDI disability benefits and early retirement at the same time?",
         triggers=["disability", "retirement"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="complex",
         query="If I keep working while collecting Social Security, how does that affect my taxes and monthly payment?",
         triggers=["earnings", "tax"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="complex",
         query="Explain the difference between delayed retirement credits and early retirement reductions",
         triggers=["delayed", "reduced"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="complex",
         query="What happens to my Social Security disability benefits when I reach full retirement age?",
         triggers=["disability", "retirement"],
         expected_tool="SearchKB", expected_grd=True),
]

# ── Category 3: Multi-turn / context-dependent ────────────────────────────────
# These are interleaved PAIRS. Set followup_of to the 0-based index within this list.
MULTI_TURN = [
    # Pair 1: retirement age -> early retirement follow-up
    dict(category="multi_turn",
         query="What is the full retirement age?",
         triggers=["retirement age", "67"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=None),
    dict(category="multi_turn",
         query="And what about the earlier option?",
         triggers=["62", "reduced"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=0),   # refers to previous turn

    # Pair 2: SSDI -> appeal timeline follow-up
    dict(category="multi_turn",
         query="Tell me about SSDI benefits",
         triggers=["disability", "ssdi"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=None),
    dict(category="multi_turn",
         query="How long does an appeal take for that?",
         triggers=["appeal", "reconsideration"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=2),

    # Pair 3: Medicare -> spouse follow-up
    dict(category="multi_turn",
         query="When can I enroll in Medicare?",
         triggers=["medicare", "65"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=None),
    dict(category="multi_turn",
         query="What about my spouse, does the same apply to them?",
         triggers=["spouse", "medicare"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=4),

    # Pair 4: survivor benefits -> minor-children follow-up
    dict(category="multi_turn",
         query="Are survivor benefits paid to a widow with children?",
         triggers=["survivor", "children"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=None),
    dict(category="multi_turn",
         query="And what age limit applies to those children?",
         triggers=["age", "18"],
         expected_tool="SearchKB", expected_grd=True,
         followup_of=6),

    # Pair 5: policy lookup -> memory follow-up
    dict(category="multi_turn",
         query="Show me section 2",
         triggers=["ssdi", "disability"],
         expected_tool="GetPolicy", expected_grd=True,
         followup_of=None),
    dict(category="multi_turn",
         query="What was the main rule in that section?",
         triggers=["ssdi", "disability"],
         expected_tool="GetPolicy", expected_grd=True,
         followup_of=8),
]

# ── Category 4: Edge cases / ambiguous ───────────────────────────────────────
EDGE = [
    dict(category="edge_case",
         query="",
         triggers=None,
         expected_tool=None, expected_grd=False),
    dict(category="edge_case",
         query="disability",
         triggers=["disability"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="edge_case",
         query="RETIRMENT BNEFITS ELIGIBILTY",                   # all-caps + typos
         triggers=["retirement", "benefits"],
         expected_tool="SearchKB", expected_grd=True),
    dict(category="edge_case",
         query="help",                                           # single-word vague
         triggers=None,
         expected_tool=None, expected_grd=False),
    dict(category="edge_case",
         query="asdfgh qwerty zxcvb",                            # gibberish
         triggers=None,
         expected_tool=None, expected_grd=False),
]

# ── Category 5: Policy-related (exercises the section mapper) ────────────────
POLICY = [
    dict(category="policy",
         query="What does section 3 say?",
         triggers=["survivor", "spouse"],
         expected_tool="GetPolicy", expected_grd=True),
    dict(category="policy",
         query="Show me the rule on SSI",
         triggers=["ssi", "needs-based"],
         expected_tool="GetPolicy", expected_grd=True),
    dict(category="policy",
         query="Cite the policy for Medicare enrollment",
         triggers=["medicare", "enrolment"],
         expected_tool="GetPolicy", expected_grd=True),
    dict(category="policy",
         query="Which regulation covers work credits and retirement?",
         triggers=["credits", "retirement"],
         expected_tool="GetPolicy", expected_grd=True),
    dict(category="policy",
         query="Give me the policy statement on disability insurance",
         triggers=["disability", "ssdi"],
         expected_tool="GetPolicy", expected_grd=True),
]


def all_test_questions():
    """Yield the 30 questions in presentation order: simple → complex → multi-turn
    → edge → policy."""
    for pool in (SIMPLE, COMPLEX, MULTI_TURN, EDGE, POLICY):
        for item in pool:
            yield item


def by_category():
    return {
        "simple":      SIMPLE,
        "complex":     COMPLEX,
        "multi_turn":  MULTI_TURN,
        "edge_case":   EDGE,
        "policy":      POLICY,
    }


if __name__ == "__main__":
    from collections import Counter
    items = list(all_test_questions())
    print(f"Total: {len(items)} questions")
    print(f"By category: {Counter(i['category'] for i in items)}")
    for i, q in enumerate(items):
        mark = ""
        if q.get("followup_of") is not None:
            mark = f" [follow-up of #{q['followup_of']}]"
        print(f"  {i:2d}. [{q['category']:<11}]{mark} {q['query'][:80]!r}")
