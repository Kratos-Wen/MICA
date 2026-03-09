
"""Fault Handler agent: common problems and fixes from KB."""
from __future__ import annotations
from typing import Dict, Any, List

def answer(context: Dict[str,Any], kb: Dict[str,Any]) -> str:
    """
    Provide troubleshooting pointers from KB.

    Args:
        context: Dict with 'detections'.
        kb: Knowledge base.

    Returns:
        Text response.
    """
    present = set(context.get("detections", []))
    lines=[]
    for comp in kb.get("components", []):
        cname = str(comp.get("name","")).strip().lower()
        if cname in present:
            probs = comp.get("problems", []) or []
            if probs:
                lines.append(f"* {comp.get('name')}: " + " | ".join(probs))
    if not lines: return "No known issues listed for visible components."
    return "Troubleshooting:\n" + "\n".join(lines)
