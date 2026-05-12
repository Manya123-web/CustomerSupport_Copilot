"""
demo/app.py
-----------
Stand-alone FastAPI demo with per-session conversation memory.

Run
---
    uvicorn demo.app:app --host 0.0.0.0 --port 8000

Query
-----
    # First turn — server creates session automatically
    curl -s -X POST http://localhost:8000/query \
        -H 'content-type: application/json' \
        -d '{"query": "What is the retirement age?", "session_id": "user-42"}'

    # Follow-up — same session_id, pronoun resolution + dialogue prefix
    curl -s -X POST http://localhost:8000/query \
        -H 'content-type: application/json' \
        -d '{"query": "And what about the early option?", "session_id": "user-42"}'

Session memory is in-process (not persisted). Restarting the server
clears all sessions — see training/memory.py for the deliberate reasoning.

Endpoints
---------
    GET  /health                      health check
    POST /query                       main inference endpoint
    POST /session/{sid}/reset         clear a session's memory
    GET  /session/{sid}               inspect a session's stored turns
"""
from __future__ import annotations
import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager
from functools import partial
from typing import Dict, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.config import load_config
from utils.schema import AgentResponse
from training.evaluation import build_system
from training.agent import run_agent
from training.memory import ConversationMemory

logger = logging.getLogger(__name__)


class QueryBody(BaseModel):
    query: str
    top_k: Optional[int] = None
    session_id: Optional[str] = None        # omit for stateless (single-turn) usage


_SYSTEM: dict = {}
_SESSIONS: Dict[str, ConversationMemory] = {}
_DEFAULT_MAX_TURNS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_path = os.environ.get("COPILOT_CONFIG", "config/full.yaml")
    print(f"[demo] loading {cfg_path}")
    _SYSTEM["sys"] = build_system(load_config(cfg_path))
    print("[demo] ready")
    yield
    _SYSTEM.clear()
    _SESSIONS.clear()


app = FastAPI(title="Customer Support Copilot", lifespan=lifespan)

# Allow browser clients to receive responses across origins. Tighten
# `allow_origins` in production deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static UI ────────────────────────────────────────────────────────────────
# A minimal single-page HTML client lives in demo/static/. Mounting under
# /static and serving index.html at "/" gives a browser-friendly UI without
# changing any of the API endpoints below.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the HTML UI when present; fall back to the legacy JSON probe."""
    if _INDEX_FILE.is_file():
        return FileResponse(str(_INDEX_FILE))
    return JSONResponse({"status": "ok",
                         "message": "Customer Support Copilot API is running"})


@app.get("/api")
def api_root():
    """Legacy JSON health probe (was previously served at '/')."""
    return {"status": "ok",
            "message": "Customer Support Copilot API is running"}


@app.get("/health")
def health():
    return {"status": "ok",
            "ready":    bool(_SYSTEM.get("sys")),
            "sessions": len(_SESSIONS)}


def _get_or_create_memory(session_id: Optional[str]) -> Optional[ConversationMemory]:
    """Returns a session's memory, creating it lazily. None for stateless calls."""
    if not session_id:
        return None
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ConversationMemory(max_turns=_DEFAULT_MAX_TURNS)
    return _SESSIONS[session_id]


@app.post("/query", response_model=AgentResponse)
async def query(body: QueryBody):
    sys = _SYSTEM.get("sys")
    if not sys:
        raise HTTPException(503, "system not ready")
    cfg = dict(sys["cfg"])
    if body.top_k:
        cfg = {**cfg, "agent": {**cfg.get("agent", {}), "top_k": body.top_k}}

    memory = _get_or_create_memory(body.session_id)
    try:
        # run_agent is synchronous and blocks for several seconds during
        # model inference. Offload to a worker thread so the event loop
        # stays responsive and the response actually reaches the client.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(run_agent,
                    body.query,
                    retriever=sys["retriever"],
                    llm_fn=sys["llm_fn"],
                    reranker=sys["reranker"],
                    cfg=cfg,
                    learned_router=sys.get("learned_router"),
                    memory=memory),
        )
        # Serialise explicitly so FastAPI does not silently emit an empty
        # body when the response model serialisation has any rough edges.
        return JSONResponse(content=response.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("[/query] failed")
        raise HTTPException(status_code=500, detail=f"{e}\n{tb}")


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    if session_id in _SESSIONS:
        _SESSIONS[session_id].clear()
        return {"session_id": session_id, "status": "cleared"}
    raise HTTPException(404, f"session {session_id!r} not found")


@app.get("/session/{session_id}")
def get_session(session_id: str):
    mem = _SESSIONS.get(session_id)
    if mem is None:
        raise HTTPException(404, f"session {session_id!r} not found")
    return {"session_id": session_id,
            "turns":      len(mem),
            "history":    [{"user": u, "assistant": a} for u, a in mem.as_list()]}
