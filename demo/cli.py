"""
demo/cli.py
-----------
Tiny CLI for interactive demo without spinning up FastAPI. Handy for
screen recordings of the project demonstrating tool-call traces.

Usage
-----
    python -m demo.cli --config config/full.yaml
"""
from __future__ import annotations
import argparse
import json

from utils.config import load_config
from training.evaluation import build_system
from training.agent import run_agent
from training.memory import ConversationMemory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/full.yaml")
    ap.add_argument("--memory-turns", type=int, default=4,
                     help="Number of past turns to retain. 0 = memory off.")
    args = ap.parse_args()
    sys = build_system(load_config(args.config))

    # One memory object for the whole CLI session. Type `/reset` to clear.
    memory = ConversationMemory(max_turns=args.memory_turns) if args.memory_turns else None

    print("\n=== Copilot CLI (Ctrl-C to exit, /reset to clear memory) ===\n")
    while True:
        try:
            q = input("▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/reset":
            if memory is not None:
                memory.clear()
                print("(memory cleared)\n")
            continue
        if q == "/memory":
            if memory is not None:
                print(f"(memory has {len(memory)} turns)")
                for i, (u, a) in enumerate(memory.as_list(), 1):
                    print(f"  {i}. User: {u[:80]}")
                    print(f"     Asst: {a[:80]}")
                print()
            continue
        resp = run_agent(q, retriever=sys["retriever"],
                          llm_fn=sys["llm_fn"],
                          reranker=sys["reranker"],
                          cfg=sys["cfg"],
                          learned_router=sys.get("learned_router"),
                          memory=memory)
        print("\n--- ANSWER ---")
        print(resp.final_answer)
        print("\n--- CITATIONS ---", resp.citations)
        print("grounding =", round(resp.grounding_score, 4),
              " id =", round(resp.id_score, 4),
              " latency =", round(resp.latency_ms, 1), "ms")
        if resp.ticket:
            print("ticket =", resp.ticket.get("ticket_id"))
        print("--- TRACE ---")
        for s in resp.tool_trace:
            print(" ", json.dumps(s)[:120])
        print()


if __name__ == "__main__":
    main()
