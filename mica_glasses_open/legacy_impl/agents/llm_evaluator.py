from __future__ import annotations
from typing import Dict, Any, List
import json
from mica_glasses_open.legacy_impl.core.llm import LLM

class LLMEvaluatorAgent:
    """LLM-based evaluator that selects the best candidate (for SharedMemory)."""
    def __init__(self, llm: LLM, prompts: Dict[str, Any], lang: str = "en"):
        self.llm = llm
        self.p = prompts.get("evaluator", {})
        self.lang = lang

    def select(self, question: str, step_id: str, focus: str, detections: str,
               candidates: List[str]) -> Dict[str, Any]:
        sys_p = self.p.get("system","")
        cand_block = "\n".join([f"[{i}] {t}" for i,t in enumerate(candidates)])
        usr_p = self.p.get("user","").format(
            question=question, step_id=step_id, focus=focus,
            detections=detections, candidates=cand_block, lang=self.lang
        )
        out = self.llm.generate(sys_p, usr_p)
        try:
            data = json.loads(out)
            if isinstance(data, dict) and "best_index" in data:
                return data
        except Exception:
            pass
        # fallback: pick shortest non-empty
        best = 0
        non_empty = [i for i,t in enumerate(candidates) if t.strip()]
        if non_empty:
            best = min(non_empty, key=lambda i: len(candidates[i]))
        return {"best_index": best, "reason": "fallback-shortest"}
