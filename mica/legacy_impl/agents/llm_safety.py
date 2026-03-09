"""LLM-based safety checker."""
from __future__ import annotations
from typing import Dict, Any, List
import json
from mica.legacy_impl.agents.llm_base import LLMAgent

class LLMSafetyChecker(LLMAgent):
    def __init__(self, llm, prompts, lang="en"):
        super().__init__(llm, prompts, "SafetyChecker", lang)

    def audit(self, step_id: str, focus: str, detections: str,
              kb_safety: str, assistant_answer: str) -> Dict[str, Any]:
        payload = dict(step_id=step_id, focus=focus, detections=detections,
                       kb_safety=kb_safety or "n/a", assistant_answer=assistant_answer)
        out = self.answer(**payload)  # LLM generate
        try:
            data = json.loads(out)
            if not isinstance(data, dict): raise ValueError
            # normalize
            v = (data.get("verdict","") or "").upper()
            data["verdict"] = "SAFE" if "SAFE" in v else ("UNSAFE" if "UNSAFE" in v else "UNSURE")
            data["reasons"] = data.get("reasons", []) or []
            data["fix"] = data.get("fix", []) or []
            return data
        except Exception:
            # tolerant fallback if model didn't return JSON
            return {"verdict": "UNSURE", "reasons": [out[:400]], "fix": []}
