
"""Maintenance Advisor agent: maintenance and care from KB."""
from __future__ import annotations
from typing import Dict, Any, List

def answer(context: Dict[str,Any], kb: Dict[str,Any]) -> str:
    """
    Provide maintenance advice for visible components.

    Args:
        context: Dict with 'detections'.
        kb: Knowledge base.

    Returns:
        Textual advice.
    """
    present = set(context.get("detections", []))
    lines=[]
    for comp in kb.get("components", []):
        cname = str(comp.get("name","")).strip().lower()
        if cname in present:
            maint = comp.get("maintenance", []) or []
            if maint:
                lines.append(f"* {comp.get('name')}: " + " | ".join(maint))
    if not lines: return "No maintenance notes found for current components."
    return "Maintenance notes:\n" + "\n".join(lines)
