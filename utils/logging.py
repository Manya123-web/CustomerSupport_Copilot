"""
utils/logging.py
----------------
Lightweight JSON-line logger. Every run dumps metrics + metadata to
experiments/results/<run_id>.json so reports can cite stable numbers
instead of reading from notebook cell output.
"""
from __future__ import annotations
import json
import os
import subprocess
import time
import uuid
from typing import Any, Dict


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def write_run(results_dir: str, config_name: str, metrics: Dict[str, Any],
              extra: Dict[str, Any] | None = None) -> str:
    os.makedirs(results_dir, exist_ok=True)
    run_id = f"{config_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    payload = {
        "run_id":      run_id,
        "config":      config_name,
        "git_sha":     _git_sha(),
        "timestamp":   int(time.time()),
        "metrics":     metrics,
        **(extra or {}),
    }
    out_path = os.path.join(results_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def latest_run(results_dir: str, config_name: str) -> Dict[str, Any] | None:
    if not os.path.isdir(results_dir):
        return None
    files = [f for f in os.listdir(results_dir)
             if f.startswith(f"{config_name}_") and f.endswith(".json")]
    if not files:
        return None
    files.sort()
    with open(os.path.join(results_dir, files[-1])) as f:
        return json.load(f)
