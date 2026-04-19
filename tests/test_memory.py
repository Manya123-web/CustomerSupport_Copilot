"""tests/test_memory.py — verifies session memory behaviour."""
from training.memory import ConversationMemory


def test_add_and_length():
    m = ConversationMemory(max_turns=3)
    assert len(m) == 0
    m.add_turn("hi", "hello")
    assert len(m) == 1
    m.add_turn("how are you", "fine")
    assert len(m) == 2


def test_buffer_bounded_to_max_turns():
    m = ConversationMemory(max_turns=2)
    m.add_turn("q1", "a1")
    m.add_turn("q2", "a2")
    m.add_turn("q3", "a3")
    m.add_turn("q4", "a4")
    assert len(m) == 2
    # Only the two most-recent turns are retained
    turns = m.as_list()
    assert turns == [("q3", "a3"), ("q4", "a4")]


def test_clear():
    m = ConversationMemory()
    m.add_turn("q", "a")
    m.clear()
    assert len(m) == 0
    assert m.as_list() == []


def test_reference_resolution_prepends_prev_turn():
    m = ConversationMemory(max_turns=4)
    m.add_turn("What is the retirement age?",
                "Full retirement age is 67.")
    expanded = m.resolve_reference("And the earlier option?")
    # Short + contains "and" + "earlier" -> prepends previous user turn
    assert "retirement age" in expanded
    assert "earlier" in expanded


def test_reference_resolution_leaves_standalone_alone():
    """A self-contained question should NOT be expanded."""
    m = ConversationMemory(max_turns=4)
    m.add_turn("What is the retirement age?",
                "Full retirement age is 67.")
    # No reference word and long enough — should pass through untouched
    long_standalone = (
        "How do I appeal a denied Social Security Disability Insurance claim "
        "if my condition has worsened since initial filing?"
    )
    assert m.resolve_reference(long_standalone) == long_standalone


def test_reference_resolution_empty_memory():
    m = ConversationMemory()
    q = "and what about that?"
    # No history to resolve against — returns unchanged
    assert m.resolve_reference(q) == q


def test_history_prompt_empty():
    m = ConversationMemory()
    assert m.history_prompt() == ""


def test_history_prompt_format():
    m = ConversationMemory(max_turns=2)
    m.add_turn("first question", "first answer")
    m.add_turn("second question", "second answer")
    prompt = m.history_prompt()
    assert "Previous conversation:" in prompt
    assert "User: first question" in prompt
    assert "User: second question" in prompt
    assert "Assistant: first answer" in prompt
    assert prompt.endswith("\n\n")


def test_history_truncates_long_assistant_responses():
    m = ConversationMemory()
    long_answer = "x" * 500
    m.add_turn("q", long_answer)
    prompt = m.history_prompt()
    # 200 char cap per assistant turn + ellipsis
    assert "xxx..." in prompt
    assert len([ln for ln in prompt.split("\n") if "xxxxxxx" in ln][0]) < 250
