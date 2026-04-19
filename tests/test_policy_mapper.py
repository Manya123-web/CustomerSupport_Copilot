"""tests/test_policy_mapper.py — verifies the query→section_id mapper."""
from training.agent import _map_query_to_policy_section


def test_explicit_section_reference():
    assert _map_query_to_policy_section("Show me section 3")     == "section_3"
    assert _map_query_to_policy_section("Sec 5")                 == "section_5"
    assert _map_query_to_policy_section("What's in section_2?")  == "section_2"


def test_topic_keyword_mapping():
    # Each section has distinctive topic keywords; the mapper picks the
    # longest match so "disability benefits" -> section_2 (not section_1).
    assert _map_query_to_policy_section("rule for disability benefits")   == "section_2"
    assert _map_query_to_policy_section("policy on survivor spouse")      == "section_3"
    assert _map_query_to_policy_section("SSI eligibility rules")           == "section_4"
    assert _map_query_to_policy_section("Medicare Part B policy")          == "section_5"
    assert _map_query_to_policy_section("rule about work credits")         == "section_1"


def test_no_match_returns_none():
    # Queries that don't hit any keyword should return None so the caller
    # can fall back to KB retrieval
    assert _map_query_to_policy_section("completely unrelated nonsense query") is None
    assert _map_query_to_policy_section("")                                    is None


def test_mapper_is_case_insensitive():
    assert _map_query_to_policy_section("DISABILITY BENEFITS policy") == "section_2"
    assert _map_query_to_policy_section("MEDICARE") == "section_5"
