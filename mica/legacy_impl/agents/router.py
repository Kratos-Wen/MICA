"""Query router and safety checker."""
from __future__ import annotations
from typing import Dict, Any
from mica.legacy_impl.core.llm import LLM

def route(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["fault","error","malfunction","broken"]):
        return "fault"
    if any(k in q for k in ["maintain","maintenance","servicing","service","lubricat","保养","维护","润滑"]):
        return "maintenance"
    if any(k in q for k in ["part","component","what is","which","零件","部件"]):
        return "parts"
    if any(k in q for k in ["step","procedure","how to","next","哪一步","步骤"]):
        return "assembly"
    return "general"

class LLMRouter:
    """
    LLM-based router using prompts.yaml->router templates.
    Falls back to heuristic route() on unexpected outputs.
    """
    LABELS = ("assembly", "parts", "maintenance", "fault", "general")

    def __init__(self, llm: LLM, router_prompts: Dict[str, str], lang: str = "en"):
        self.llm = llm
        self.system = router_prompts.get("system","")
        self.user   = router_prompts.get("user","")
        self.lang = lang

    def render(self, template: str, **kw) -> str:
        try:
            return template.format(**kw)
        except Exception:
            return template

    @staticmethod
    def normalize(label: str) -> str:
        """Map model outputs like 'maint' / 'maintenance-related' to canonical labels."""
        lab = (label or "").strip().lower()
        if "maint" in lab or "maintenance" in lab:
            return "maintenance"
        for x in ("assembly", "parts", "fault", "general"):
            if x in lab:
                return x
        return ""  # unknown

    def classify(
        self,
        step_id: str,
        focus: str,
        detections: str,
        question: str,
        previous_intent: str = "",
        previous_entities: str = ""
    ) -> str:
        """
        Classify query into one of LABELS.
        - previous_intent: last chosen label (for 'again/previous' reuse)
        - previous_entities: last entity list in order (for ordinal references)
        """
        ql = (question or "").lower()

        # --- minimal hard overrides (keep these very few and obvious) ---
        # counting -> general
        if any(k in ql for k in ["how many", "count", "数量"]):
            return "general"
        # lubrication -> maintenance
        if any(k in ql for k in ["lubricate", "lube", "grease", "oil", "润滑", "上油", "涂脂"]):
            return "maintenance"
        # 'again / previous' -> reuse previous intent when available
        if previous_intent:
            if any(k in ql for k in ["again", "previous", "same as before", "再来", "之前", "上一次"]):
                norm_prev = self.normalize(previous_intent)
                if norm_prev in self.LABELS:
                    return norm_prev

        # --- LLM prompt with history fields ---
        sys_p = self.render(
            self.system,
            step_id=step_id, focus=focus, detections=detections,
            question=question, lang=self.lang
        )
        usr_p = self.render(
            self.user,
            step_id=step_id, focus=focus, detections=detections, question=question,
            previous_intent=previous_intent or "none",
            previous_entities=previous_entities or "none",
            lang=self.lang
        )

        out = (self.llm.generate(sys_p, usr_p) or "").strip().lower()
        norm = self.normalize(out)
        if norm in self.LABELS:
            return norm

        # fallback heuristic (rare)
        return route(question)

def safety_check(answer: str, kb_safety: str) -> str:
    """
    Append a compact safety note when the draft answer lacks any safety wording.
    This is a conservative, additive check — it never removes content.

    Args:
        answer: Draft text from an agent.
        kb_safety: Aggregated safety notes (e.g., joined strings from KB).

    Returns:
        Audited answer text with a Safety Note if needed.
    """
    audit = answer or ""
    if kb_safety:
        low = audit.lower()
        if ("safety" not in low) and ("warning" not in low) and ("注意" not in low):
            audit += f"\n\nSafety Note: {kb_safety}"
    return audit
