"""
30 curated demo questions for testing the chatbot's behaviour.

Split into four categories so you can see how each system component
behaves on each kind of query:

  1. STANDARD       — answerable directly from the KB
  2. PARAPHRASED    — same intents as STANDARD but worded differently
                      (tests retrieval robustness + learned router)
  3. ACCOUNT_ISSUE  — should route to CreateTicket, not SearchKB
  4. OUT_OF_SCOPE   — KB has no good answer → CGRA should escalate

Use `all_demo_questions()` for an iterable, or `demo_by_category()` for
a dict keyed by category.
"""

STANDARD = [
    ("What documents do I need to apply for Social Security retirement benefits?",
     ["retirement", "documents", "apply", "social security"]),
    ("How many work credits do I need to qualify for Social Security?",
     ["work credits", "40 credits", "qualify"]),
    ("When can I start receiving reduced early retirement benefits?",
     ["early retirement", "age 62", "reduced"]),
    ("How are survivor benefits paid to a widow with minor children?",
     ["survivor", "spouse", "children", "minor"]),
    ("What is the difference between SSI and SSDI?",
     ["ssi", "supplemental security income", "ssdi"]),
    ("How do I appeal a denied disability claim?",
     ["appeal", "denied", "reconsideration"]),
    ("At what age should I enroll in Medicare Part A and Part B?",
     ["medicare", "enroll", "part a", "part b", "65"]),
    ("How is my Social Security retirement benefit amount calculated?",
     ["benefit", "amount", "calculated", "formula"]),
]

PARAPHRASED = [
    ("I'm turning 62 next year — when's the earliest I can start getting checks?",
     ["early retirement", "age 62"]),
    ("My husband passed away last month. Am I entitled to anything?",
     ["survivor", "spouse", "deceased"]),
    ("Can my ex-wife claim benefits based on my work record?",
     ["divorced", "former spouse", "work record"]),
    ("I'm a stay-at-home parent — do I qualify for anything on my own record?",
     ["qualify", "eligibility", "record"]),
    ("What happens to my retirement payment if I move to Canada?",
     ["abroad", "outside", "foreign"]),
    ("I work part-time and collect Social Security. Is there a limit on my earnings?",
     ["earnings test", "limit", "working"]),
    ("Are my benefits subject to federal income tax?",
     ["tax", "taxable", "irs"]),
    ("How long does a disability review take to process?",
     ["disability", "process", "review"]),
]

ACCOUNT_ISSUE = [
    ("I can't log into my account",                         None),
    ("I forgot my password and can't reset it",             None),
    ("My online portal is throwing an error when I sign in", None),
    ("I am locked out of my Social Security account",       None),
    ("The website says access denied every time I try",     None),
    ("My account shows wrong dependent information",        None),
    ("Something is broken with my profile page",            None),
]

OUT_OF_SCOPE = [
    # These genuinely have no good answer in MultiDoc2Dial; CGRA should
    # trigger the web scraper and, if still weak, escalate with a ticket.
    ("What is the tax treaty between the US and France for expatriates?", None),
    ("Can I invest my Social Security contributions in cryptocurrency?",   None),
    ("What's the weather forecast for my local SSA office tomorrow?",      None),
    ("Who is the current Commissioner of Social Security Administration?", None),
    ("Can I use my SSN as a login for third-party banking apps?",          None),
    ("Is there a Social Security benefit for caring for a pet?",           None),
    ("Can I transfer my Social Security credits to my spouse in India?",   None),
]


def demo_by_category() -> dict:
    return {
        "STANDARD":      STANDARD,
        "PARAPHRASED":   PARAPHRASED,
        "ACCOUNT_ISSUE": ACCOUNT_ISSUE,
        "OUT_OF_SCOPE":  OUT_OF_SCOPE,
    }


def all_demo_questions():
    for cat, qs in demo_by_category().items():
        for q, triggers in qs:
            yield {"category": cat, "query": q, "triggers": triggers}


if __name__ == "__main__":
    for i, item in enumerate(all_demo_questions(), 1):
        print(f"{i:2d}. [{item['category']:<13}] {item['query']}")
