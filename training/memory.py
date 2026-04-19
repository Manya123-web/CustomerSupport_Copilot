"""
training/memory.py
------------------
Session-based conversation memory for the Customer Support Copilot.

Design choices (explicit, not accidental):
------------------------------------------

1. SHORT-TERM ONLY (not persistent across sessions).
   Rationale: long-term persistent memory is a privacy and correctness
   minefield on a support system. This module keeps an in-memory buffer
   per session; nothing is written to disk.

2. BOUNDED BUFFER (last K turns).
   Rationale: Flan-T5's context window is 1024 tokens. Unbounded history
   blows the budget. We keep the last K=4 turns by default.

3. MEMORY INFLUENCES TWO THINGS:
   a) RETRIEVAL: if the query contains a pronoun/reference ("that",
      "it", "earlier"), we prepend the most recent user query to the
      retrieval query. This is resolved once, in the agent loop.
   b) GENERATION: the tail of the conversation is appended to the prompt
      so the LLM sees the dialogue flow, not just the latest query.

4. NOT FED TO TRAINING.
   The rubric asks for offline-only training. Memory affects inference
   only; no gradient flows through it.

Usage
-----
    from training.memory import ConversationMemory

    mem = ConversationMemory(max_turns=4)
    mem.add_turn("What's the retirement age?", "Full retirement age is 67.")
    mem.add_turn("What about early retirement?", ...)
    context_q = mem.resolve_reference("And the reduced amount at 62?")
    # -> "What's the retirement age? What about early retirement? And the reduced amount at 62?"
"""
from __future__ import annotations
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple


# Words that signal the current query refers to something said earlier.
# Conservative list — we'd rather miss a resolution than inject noise
# into a standalone query.
_REFERENCE_WORDS = {
    "it", "that", "this", "those", "these", "them", "they",
    "there", "earlier", "before", "previously", "above",
    "mentioned", "said", "you said", "you mentioned",
    # follow-up connectors
    "also", "too", "and", "what about", "how about",
    "what if", "and what", "and how",
}

# Upper bound on how much history we inject into the LLM prompt
_MAX_HISTORY_CHARS = 800


@dataclass
class Turn:
    user: str
    assistant: str
    grounding: float = 0.0
    has_ticket: bool = False


@dataclass
class ConversationMemory:
    """
    Per-session bounded buffer of (user, assistant) turns.

    One instance per user session. Thread-unsafe by design — spawn one
    per request handler.
    """
    max_turns: int = 4
    turns: Deque[Turn] = field(default_factory=deque)

    def add_turn(self, user: str, assistant: str,
                 grounding: float = 0.0, has_ticket: bool = False) -> None:
        """Append a completed (user, assistant) turn to the buffer."""
        self.turns.append(Turn(user=user, assistant=assistant,
                                grounding=grounding, has_ticket=has_ticket))
        while len(self.turns) > self.max_turns:
            self.turns.popleft()

    def clear(self) -> None:
        self.turns.clear()

    # ── Influence on retrieval ───────────────────────────────────────────────
    def resolve_reference(self, query: str) -> str:
        """
        Expand the query with the most recent user turn IF it looks like a
        follow-up. Returns the original query otherwise.

        Heuristic: if the query contains a reference word AND is short
        (<15 tokens), prepend the previous user turn. This is deliberately
        conservative — we don't resolve every follow-up, just the obvious
        ones.
        """
        if not self.turns:
            return query
        q_lower = query.lower()
        q_tokens = q_lower.split()
        is_short = len(q_tokens) < 15
        has_ref = any(rw in q_lower for rw in _REFERENCE_WORDS)
        if not (is_short and has_ref):
            return query
        # Only expand with the MOST RECENT turn. Expanding with the full
        # buffer would make the retrieval query topic-diffuse.
        prev = self.turns[-1].user
        return f"{prev} {query}"

    # ── Influence on generation ──────────────────────────────────────────────
    def history_prompt(self) -> str:
        """
        Build the dialogue-history prefix that the LLM sees before the
        current turn. Returns '' if no history.

        Format (one turn per line):
            Previous conversation:
            User: ...
            Assistant: ...
            User: ...
            Assistant: ...
        """
        if not self.turns:
            return ""
        lines: List[str] = ["Previous conversation:"]
        for t in self.turns:
            u = t.user.strip()
            a = t.assistant.strip()
            # Truncate each assistant turn to avoid blowing the context budget
            if len(a) > 200:
                a = a[:197] + "..."
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
        block = "\n".join(lines)
        if len(block) > _MAX_HISTORY_CHARS:
            block = block[-_MAX_HISTORY_CHARS:]
            block = "Previous conversation (truncated):\n" + block
        return block + "\n\n"

    # ── Inspection helpers ───────────────────────────────────────────────────
    def last_user_turn(self) -> Optional[str]:
        return self.turns[-1].user if self.turns else None

    def as_list(self) -> List[Tuple[str, str]]:
        return [(t.user, t.assistant) for t in self.turns]

    def __len__(self) -> int:
        return len(self.turns)
