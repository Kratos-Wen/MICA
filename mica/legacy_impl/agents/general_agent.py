
"""General Agent: fallback responses mixing parts info and current step."""
from __future__ import annotations
from typing import Dict, Any

def answer(context: Dict[str,Any], kb: Dict[str,Any]) -> str:
    """
    Provide a general summary of the current scene.

    Args:
        context: Dict with 'detections' and 'step_id'.
        kb: Knowledge base.

    Returns:
        Text summary.
    """
    dets = context.get("detections", [])
    step_id = context.get("step_id", "Unknown")
    return f"Current step: {step_id}. Visible components: {', '.join(sorted(set(dets))) or 'none'}."
