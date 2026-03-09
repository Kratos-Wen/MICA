"""LLM-backed specialized agents."""
from __future__ import annotations
from typing import Dict, Any
from mica_glasses_open.legacy_impl.agents.llm_base import LLMAgent

class LLMAssemblyGuide(LLMAgent):
    def __init__(self, llm, prompts, lang="en"):
        super().__init__(llm, prompts, "AssemblyGuide", lang)

class LLMPartsAdvisor(LLMAgent):
    def __init__(self, llm, prompts, lang="en"):
        super().__init__(llm, prompts, "PartsAdvisor", lang)

class LLMMaintenanceAdvisor(LLMAgent):
    def __init__(self, llm, prompts, lang="en"):
        super().__init__(llm, prompts, "MaintenanceAdvisor", lang)

class LLMFaultHandler(LLMAgent):
    def __init__(self, llm, prompts, lang="en"):
        super().__init__(llm, prompts, "FaultHandler", lang)

class LLMGeneralAgent(LLMAgent):
    def __init__(self, llm, prompts, lang="en"):
        super().__init__(llm, prompts, "GeneralAgent", lang)
