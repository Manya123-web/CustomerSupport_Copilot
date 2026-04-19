"""
utils/schema.py
---------------
Strict tool-call schemas. Every tool invocation is validated through one of
these pydantic models before execution. This replaces the previous version's
implicit dict contract and is what makes the "structured output" claim
auditable.
"""
from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Arguments per tool (what the agent produces) ─────────────────────────────
class SearchKBArgs(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class GetPolicyArgs(BaseModel):
    section_id: str = Field(..., min_length=1)
    query: str = ""


class CreateTicketArgs(BaseModel):
    summary: str = Field(..., min_length=1)
    category: Literal["account", "payments", "general"] = "general"
    severity: Literal["low", "medium", "high"] = "medium"


# ── Outer envelope that the agent emits per step ─────────────────────────────
class ToolCall(BaseModel):
    tool: Literal["SearchKB", "GetPolicy", "CreateTicket"]
    arguments: dict

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, v, info):
        """Cross-validate arguments against the declared tool."""
        tool = info.data.get("tool")
        mapping = {
            "SearchKB":     SearchKBArgs,
            "GetPolicy":    GetPolicyArgs,
            "CreateTicket": CreateTicketArgs,
        }
        if tool in mapping:
            mapping[tool](**v)       # raises if shape is wrong
        return v


# ── Chunk returned by retrieval ──────────────────────────────────────────────
class Chunk(BaseModel):
    chunk_id: str
    doc_id:   str
    text:     str
    score:    float = 0.0
    cos_score: Optional[float] = None
    lex_score: Optional[float] = None
    ce_score:  Optional[float] = None
    source:    str = "kb"


# ── Final agent response (stable API contract for demo.py) ───────────────────
class AgentResponse(BaseModel):
    final_answer: str
    citations:    List[str] = []
    grounding_score: float = 0.0
    id_score:     float = 0.0
    ticket:       Optional[dict] = None
    tool_trace:   List[dict] = []
    latency_ms:   float = 0.0
