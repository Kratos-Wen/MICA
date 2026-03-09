"""LLM JSON helper to robustly extract structured outputs."""
from __future__ import annotations
from typing import Any, Dict
import json
from mica.legacy_impl.core.llm import LLM

def ask_json(llm: LLM, system: str, user: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call LLM and parse JSON. If parsing fails, return fallback.
    """
    out = llm.generate(system, user)
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return fallback
