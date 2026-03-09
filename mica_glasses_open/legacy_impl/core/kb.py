"""Knowledge base loading, alias mapping, and RAG helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import json

def load_kb(path: str) -> Dict[str, Any]:
    """
    Load the KB JSON file.

    Args:
        path: Path to the KB JSON.

    Returns:
        Dict[str, Any]: KB dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KB not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def build_alias_map(kb: Dict[str, Any]) -> Dict[str, str]:
    """
    Build alias→canonical map (lowercased).

    Args:
        kb: The KB dictionary.

    Returns:
        Dict[str, str]: mapping from alias to canonical.
    """
    m: Dict[str,str] = {}
    aliases = kb.get("aliases", {}) or {}
    for canon, syns in aliases.items():
        c = canon.strip().lower()
        m[c] = c
        for s in syns or []:
            m[str(s).strip().lower()] = c
    # also include component names themselves
    for comp in kb.get("components", []):
        c = str(comp.get("name","")).strip().lower()
        if c: m[c] = c
    return m

# def rag_fields_for_component(c: Dict[str, Any]) -> Dict[str, List[str]]:
#     """
#     Extract RAG-usable fields from a component record.
#
#     Args:
#         c: Component dict.
#
#     Returns:
#         Dict[str, List[str]]: fields with lists of strings.
#     """
#     return {
#         "key_features": c.get("key_features", []) or [],
#         "tools": c.get("tools", []) or [],
#         "safety": c.get("safety", []) or [],
#         "parts_list": c.get("parts_list", []) or [],
#         "assembly_steps": c.get("assembly_steps", []) or [],
#         "maintenance": c.get("maintenance", []) or [],
#         "problems": c.get("problems", []) or []
#     }
def rag_fields_for_component(c: Dict[str, Any]) -> Dict[str, List[str]]:
    def pick(*keys):
        for k in keys:
            v = c.get(k)
            if isinstance(v, list): return v
        return []
    return {
        "key_features": pick("key_features", "Key Features"),
        "tools":        pick("tools", "Tools and Equipment"),
        "safety":       pick("safety", "Assembly Safety Instructions"),
        "parts_list":   pick("parts_list", "Parts List"),
        "assembly_steps": pick("assembly_steps", "Assembly Steps"),
        "maintenance":  pick("maintenance", "Maintenance and Care"),
        "problems":     pick("problems", "Common Problems and Solutions"),
    }
