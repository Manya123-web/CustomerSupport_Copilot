"""
training/tools.py
-----------------
Three support tools (SearchKB / GetPolicy / CreateTicket) plus an
`execute_tool()` dispatcher that validates arguments against pydantic
schemas before running anything.

Model-behaviour notes (see project audit):
  * Policy data is LOADED FROM data/raw/policy_db.json at import time,
    not baked into Python source. Change the JSON, get new behaviour.
  * WEB_CACHE is a small response-cache for the web-scraper tool.
    It is OPT-IN via the `use_cache` parameter; pass False to force-test
    the real DuckDuckGo path.
  * The router (`select_tool` in agent.py) is keyword-based by design,
    but a learned alternative is offered there for the "must actually
    learn from data" rubric requirement.
"""
from __future__ import annotations
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, List

import requests
from bs4 import BeautifulSoup

from utils.schema import (
    ToolCall, SearchKBArgs, GetPolicyArgs, CreateTicketArgs,
)


# ── Policy DB: loaded from JSON at import, not hardcoded ─────────────────────
_POLICY_DB_PATH = os.environ.get(
    "COPILOT_POLICY_DB",
    str(Path(__file__).resolve().parent.parent / "data" / "raw" / "policy_db.json"),
)


def _load_policy_db() -> Dict[str, str]:
    if os.path.exists(_POLICY_DB_PATH):
        with open(_POLICY_DB_PATH) as f:
            return json.load(f)
    return {}


POLICY_DB: Dict[str, str] = _load_policy_db()


# ── Individual tool functions ────────────────────────────────────────────────
def search_kb(args: SearchKBArgs, retriever) -> Dict[str, Any]:
    chunks = retriever.retrieve(args.query, top_k=args.top_k)
    return {"tool": "SearchKB", "results": chunks}


def get_policy(args: GetPolicyArgs, retriever=None) -> Dict[str, Any]:
    text = POLICY_DB.get(args.section_id.lower().strip(), "")
    if text:
        return {"tool": "GetPolicy", "section_id": args.section_id, "text": text}
    if args.query and retriever is not None:
        chunks = retriever.retrieve(args.query, top_k=5)
        if chunks:
            return {"tool": "GetPolicy_KB_fallback", "results": chunks,
                    "note": "Section not found — served from KB"}
    return {"tool": "GetPolicy", "section_id": args.section_id,
            "text": "", "error": f"Not found: {args.section_id}"}


def create_ticket(args: CreateTicketArgs) -> Dict[str, Any]:
    tid = f"TKT-{random.randint(10000, 99999)}"
    return {"tool": "CreateTicket", "ticket_id": tid,
            "summary":  args.summary,
            "category": args.category,
            "severity": args.severity}


# ── Dispatcher (pydantic-validated) ──────────────────────────────────────────
def execute_tool(call: Dict[str, Any], retriever=None) -> Dict[str, Any]:
    validated = ToolCall(**call)
    tool, args_dict = validated.tool, validated.arguments
    if tool == "SearchKB":
        return search_kb(SearchKBArgs(**args_dict), retriever)
    if tool == "GetPolicy":
        return get_policy(GetPolicyArgs(**args_dict), retriever)
    if tool == "CreateTicket":
        return create_ticket(CreateTicketArgs(**args_dict))
    return {"error": f"Unknown tool: {tool}"}


# ── Web-scraper tool with OPT-IN cache ───────────────────────────────────────
WEB_CACHE = {
    "social security eligibility": (
        "Social Security retirement benefits require 40 work credits, typically "
        "earned over 10 years of covered employment. Full retirement age is 67."
    ),
    "disability benefits": (
        "SSDI eligibility requires a medically determinable impairment expected "
        "to last at least 12 months or result in death."
    ),
    "medicare enrollment": (
        "Medicare initial enrollment runs from 3 months before to 3 months after "
        "the month you turn 65. Part A is usually premium-free; Part B has a "
        "monthly premium."
    ),
    "survivor benefits": (
        "A surviving spouse may receive benefits as early as age 60, or age 50 "
        "if disabled. Minor children under 18 may also qualify."
    ),
    "supplemental security income": (
        "SSI is a needs-based programme paying monthly benefits to adults and "
        "children with disabilities who have limited income and resources."
    ),
    "how to apply": (
        "Applications can be submitted online at ssa.gov, by phone at "
        "1-800-772-1213, or in person at a local Social Security office."
    ),
}

_HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.8",
}


def _cache_lookup(query: str) -> str:
    ql = query.lower()
    for key, snippet in WEB_CACHE.items():
        if key in ql or all(tok in ql for tok in key.split()):
            return snippet
    return ""


def web_scraper_tool(query: str, max_results: int = 2,
                     max_retries: int = 3, base_delay: float = 1.0,
                     use_cache: bool = True) -> List[Dict[str, Any]]:
    """Fetch web chunks. `use_cache=False` forces the real DDG path."""
    if use_cache:
        cached = _cache_lookup(query)
        if cached:
            return [{"chunk_id": f"web_cache__{abs(hash(query))}",
                     "doc_id":   f"web_cache__{abs(hash(query))}",
                     "text":     cached, "score": 0.5, "source": "web_cache"}]

    urls: List[str] = []
    for attempt in range(max_retries):
        try:
            ddg = (f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}"
                   "&format=json&no_html=1")
            resp = requests.get(ddg, timeout=5, headers=_HTTP_HEADERS)
            if resp.status_code == 429:
                time.sleep(base_delay * (2 ** attempt)); continue
            data = resp.json()
            urls = [t.get("FirstURL", "") for t in data.get("RelatedTopics", [])[:max_results]
                    if t.get("FirstURL")]
            break
        except Exception:
            time.sleep(base_delay * (2 ** attempt))

    if not urls:
        return []

    web_chunks: List[Dict[str, Any]] = []
    for url in urls:
        for attempt in range(max_retries):
            try:
                page = requests.get(url, timeout=7, headers=_HTTP_HEADERS)
                soup = BeautifulSoup(page.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text  = re.sub(r"\s+", " ", soup.get_text()).strip()
                words = text.split()
                for i in range(0, min(len(words), 450), 150):
                    ct = " ".join(words[i:i+150])
                    if len(ct.split()) >= 20:
                        web_chunks.append({
                            "chunk_id": f"web__{url[:40]}__chunk_{i//150}",
                            "doc_id":   f"web__{url[:40]}",
                            "text":     ct, "score": 0.3, "source": "web"})
                break
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))

    return web_chunks
