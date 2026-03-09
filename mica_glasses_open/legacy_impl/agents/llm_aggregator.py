from __future__ import annotations
from typing import Dict, Any, List
from mica_glasses_open.legacy_impl.core.llm import LLM

class LLMAggregatorAgent:
    """LLM-based aggregator that merges multiple answers (for CentralizedBroadcast)."""
    def __init__(self, llm: LLM, prompts: Dict[str, Any], lang: str = "en"):
        self.llm = llm
        self.p = prompts.get("aggregator", {})
        self.lang = lang

    # def merge(self, question: str, step_id: str, focus: str, detections: str,
    #           answers: List[str]) -> str:
    #     sys_p = self.p.get("system","")
    #     ans_block = "\n---\n".join(answers)
    #     usr_p = self.p.get("user","").format(
    #         question=question, step_id=step_id, focus=focus,
    #         detections=detections, answers=ans_block, lang=self.lang
    #     )
    #     out = self.llm.generate(sys_p, usr_p)
    #     return out.strip()
    def merge(self, question: str, step_id: str, focus: str, detections: str,
              answers: List[str]) -> str:
        sys_p = (self.p.get("system", "") or "") + f"\nAnswer in {self.lang}. Return only the merged text."
        lines = []
        for i, t in enumerate(answers):
            t = t.strip()
            if not t: continue
            lines.append(f"[{i}] {t}")
        ans_block = "\n---\n".join(lines[:8])  # 最多取前 8 条参与聚合，避免超长

        usr_p = (self.p.get("user", "") or "").format(
            question=question, step_id=step_id, focus=focus,
            detections=detections, answers=ans_block, lang=self.lang
        )
        out = self.llm.generate(sys_p, usr_p)
        return out.strip()
