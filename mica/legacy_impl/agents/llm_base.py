"""LLM-based agent base class using templated prompts."""
from __future__ import annotations
from typing import Dict, Any
from mica.legacy_impl.core.llm import LLM

class LLMAgent:
    """
    Render a system/user prompt template and call the LLM.
    """
    def __init__(self, llm: LLM, prompts: Dict[str, Any], agent_name: str, lang: str = "en"):
        self.llm = llm
        self.prompts = prompts.get(agent_name, {})
        self.lang = lang

    def render(self, template: str, **kw) -> str:
        try:
            return template.format(**kw)
        except Exception:
            return template

    def answer(self, **payload) -> str:
        system = self.render(self.prompts.get("system",""), **payload, lang=self.lang)
        user   = self.render(self.prompts.get("user",""), **payload, lang=self.lang)
        return self.llm.generate(system, user).strip()
