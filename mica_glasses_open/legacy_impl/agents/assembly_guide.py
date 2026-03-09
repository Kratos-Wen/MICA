
"""Assembly Guide agent: procedural assistance using KB 'assembly_steps'."""
from __future__ import annotations
from typing import Dict, Any, List

def answer(context: Dict[str,Any], kb: Dict[str,Any]) -> str:
    """
    Produce step-by-step guidance using KB 'assembly_steps'.

    Args:
        context: Dict containing 'step_id', 'detections' (canonical names), etc.
        kb: Knowledge base.

    Returns:
        Textual guidance.
    """
    present = set(context.get("detections", []))
    lines=[]
    for comp in kb.get("components", []):
        name = str(comp.get("name","")).strip().lower()
        if name in present:
            steps = comp.get("assembly_steps", []) or []
            if steps:
                lines.append(f"- {comp.get('name')}: " + " -> ".join(steps))
    if not lines:
        return "No component-specific assembly steps found in KB for current view."
    return "Assembly guidance:\n" + "\n".join(lines)
