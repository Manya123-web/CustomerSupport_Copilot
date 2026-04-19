"""
utils/config.py
---------------
Tiny YAML config loader with `extends:` inheritance so baseline.yaml and
full.yaml can re-use config.yaml without duplication.
"""
from __future__ import annotations
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config, honouring a top-level `extends: <file>` key."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(path), parent)
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)
    return cfg
