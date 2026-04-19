"""Router tests — pure keyword logic, no models, no network."""
from training.agent import select_tool, information_density


def test_router_keywords():
    assert select_tool("I can't log into my account") == "CreateTicket"
    assert select_tool("What does section 2 say?")    == "GetPolicy"
    assert select_tool("How do I apply for benefits?") == "SearchKB"
    # default bucket
    assert select_tool("hello") == "SearchKB"


def test_information_density_bounds():
    chunks = [
        {"text": "Social Security requires 40 credits earned over 10 years.",
         "ce_score": 2.0},
        {"text": "The application can be submitted online at ssa.gov.",
         "ce_score": 0.5},
    ]
    s = information_density("Social Security credits", chunks)
    assert 0.0 <= s <= 1.0

    # empty → 0
    assert information_density("anything", []) == 0.0
