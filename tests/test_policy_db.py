"""
tests/test_policy_db.py
-----------------------
Proves the POLICY_DB is loaded from data/raw/policy_db.json at import
time — i.e. the content of the policy responses is DATA, not Python code.
"""
import os
import json
import importlib
import pytest


def test_policy_db_loads_from_json(tmp_path, monkeypatch):
    # Create a fresh test JSON with a unique marker string
    tmp = tmp_path / "policy.json"
    tmp.write_text(json.dumps({"section_42": "UNIQUE_MARKER_STRING_xyz"}))

    monkeypatch.setenv("COPILOT_POLICY_DB", str(tmp))

    # Re-import the module so the env var is picked up
    import training.tools as tools
    importlib.reload(tools)

    # Now the marker should be reachable via get_policy()
    from utils.schema import GetPolicyArgs
    r = tools.get_policy(GetPolicyArgs(section_id="section_42"))
    assert r["text"] == "UNIQUE_MARKER_STRING_xyz"


def test_policy_db_reload_reflects_file_changes(tmp_path, monkeypatch):
    """If the JSON changes, reloading the module yields the new values."""
    tmp = tmp_path / "p.json"
    tmp.write_text(json.dumps({"s": "v1"}))
    monkeypatch.setenv("COPILOT_POLICY_DB", str(tmp))

    import training.tools as tools
    importlib.reload(tools)
    from utils.schema import GetPolicyArgs
    assert tools.get_policy(GetPolicyArgs(section_id="s"))["text"] == "v1"

    tmp.write_text(json.dumps({"s": "v2"}))
    importlib.reload(tools)
    assert tools.get_policy(GetPolicyArgs(section_id="s"))["text"] == "v2"
