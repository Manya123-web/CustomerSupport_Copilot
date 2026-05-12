"""Schema sanity tests — no model downloads needed."""
import pytest
from pydantic import ValidationError

from utils.schema import ToolCall, KBLookupArgs, EscalateIssueArgs, AgentResponse


def test_search_kb_args_bounds():
    KBLookupArgs(query="hello", top_k=5)
    with pytest.raises(ValidationError):
        KBLookupArgs(query="", top_k=5)
    with pytest.raises(ValidationError):
        KBLookupArgs(query="ok", top_k=0)


def test_create_ticket_enums():
    EscalateIssueArgs(summary="x", category="account", severity="high")
    with pytest.raises(ValidationError):
        EscalateIssueArgs(summary="x", category="ALIENS", severity="high")
    with pytest.raises(ValidationError):
        EscalateIssueArgs(summary="x", category="general", severity="ultra")


def test_tool_call_cross_validation():
    ok = ToolCall(tool="KBLookup", arguments={"query": "foo", "top_k": 3})
    assert ok.tool == "KBLookup"
    # Wrong argument shape for the declared tool
    with pytest.raises(ValidationError):
        ToolCall(tool="EscalateIssue", arguments={"query": "foo"})


def test_agent_response_defaults():
    r = AgentResponse(final_answer="hi")
    assert r.citations == []
    assert r.tool_trace == []
    assert r.grounding_score == 0.0
