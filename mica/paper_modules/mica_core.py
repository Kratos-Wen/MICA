"""Paper module: MICA-core."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from mica.legacy_impl.agents import parts_advisor
from mica.legacy_impl.agents.llm_agents import (
    LLMAssemblyGuide,
    LLMFaultHandler,
    LLMGeneralAgent,
    LLMMaintenanceAdvisor,
    LLMPartsAdvisor,
)
from mica.legacy_impl.agents.llm_aggregator import LLMAggregatorAgent
from mica.legacy_impl.agents.llm_evaluator import LLMEvaluatorAgent
from mica.legacy_impl.agents.llm_safety import LLMSafetyChecker
from mica.legacy_impl.agents.orchestrators import Orchestrator
from mica.legacy_impl.agents.router import LLMRouter, route, safety_check
from mica.legacy_impl.core.kb import rag_fields_for_component
from mica.legacy_impl.core.memory import Memory
from mica.runtime.kb_utils import kb_components_list, kb_index_by_name
from mica.types import FrameDecision, QAResponse


def load_prompts() -> Dict[str, Any]:
    prompt_path = Path(__file__).resolve().parents[1] / "resources" / "prompts.yaml"
    try:
        return yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"lang": "en", "router": {}, "agents": {}}


def _strip_step_prefix(text: str) -> str:
    value = str(text or "").strip()
    lowered = value.lower()
    if lowered.startswith("step "):
        parts = value.split(" ", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return value


def build_kb_snippets_for_route(route_id: str, det_names: List[str], kb_index: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    norm_names = sorted({str(name).strip().lower() for name in det_names if str(name).strip()})
    for key in norm_names:
        if key not in kb_index:
            continue
        component = kb_index[key]
        fields = rag_fields_for_component(component)
        canonical_name = component.get("name") or key

        if route_id == "assembly":
            lines.append(f"[Name] {canonical_name}")
            if fields["safety"]:
                lines.append("[Section:safety] " + " ".join(fields["safety"]))
            if fields["tools"]:
                lines.append("[Section:tools] " + "; ".join(fields["tools"]))
            if fields["parts_list"]:
                lines.append("[Section:parts_list] " + "; ".join(fields["parts_list"]))
            for index, step in enumerate(fields["assembly_steps"], start=1):
                value = _strip_step_prefix(step)
                if value:
                    lines.append(f"[Step:{index}] {value}")
        elif route_id == "parts":
            attrs = [f"[Component] {canonical_name}"]
            part_no = component.get("Part No.") or component.get("part_no") or component.get("PartNo")
            color = component.get("Color") or component.get("color")
            if part_no:
                attrs.append(f"[PartNo] {part_no}")
            if color:
                attrs.append(f"[Color] {color}")
            if fields["key_features"]:
                attrs.append("[KeyFeatures] " + " ; ".join(fields["key_features"]))
            lines.append(" ".join(attrs))
        elif route_id == "maintenance":
            sect = [f"[Component] {canonical_name}"]
            if fields["maintenance"]:
                sect.append("[Maintenance] " + " ".join(fields["maintenance"]))
            if fields["tools"]:
                sect.append("[Tools] " + "; ".join(fields["tools"]))
            if len(sect) > 1:
                lines.append(" ".join(sect))
        elif route_id == "fault":
            problems = component.get("Common Problems and Solutions", []) or component.get("problems", [])
            if isinstance(problems, list) and problems and isinstance(problems[0], dict):
                for item in problems:
                    problem = str(item.get("Problem") or item.get("problem") or "").strip()
                    solution = str(item.get("Solution") or item.get("solution") or "").strip()
                    lines.append(f"[Component] {canonical_name} [Problem] {problem} [Solution] {solution}".strip())
            elif problems:
                lines.append(f"[Component] {canonical_name} [Problems] " + " | ".join(str(item) for item in problems))
    return "\n".join(lines) if lines else "n/a"


def collect_kb_safety(det_names: List[str], kb_index: Dict[str, Dict[str, Any]]) -> str:
    notes: List[str] = []
    for name in sorted(set(det_names)):
        key = str(name).strip().lower()
        if key not in kb_index:
            continue
        safety_items = kb_index[key].get("safety", []) or kb_index[key].get("Assembly Safety Instructions", []) or []
        for item in safety_items:
            text = str(item).strip()
            if text and text not in notes:
                notes.append(text)
    return "; ".join(notes[:3]) if notes else ""


class MICACore:
    """Question answering and routing layer described in the paper."""

    def __init__(self, cfg, kb: Dict[str, Any], llm, entity_linker=None) -> None:
        self.cfg = cfg
        self.kb = kb
        self.llm = llm
        self.entity_linker = entity_linker
        self.prompts = load_prompts()
        self.lang = self.prompts.get("lang", "en")
        self.memory = Memory(max_items=20)
        self.components = kb_components_list(kb)
        self.kb_index = kb_index_by_name(self.components)

        self.router = LLMRouter(llm, self.prompts.get("router", {}), lang=self.lang)
        agents_prompt = self.prompts.get("agents", {})
        self.agents = {
            "assembly": LLMAssemblyGuide(llm, agents_prompt, lang=self.lang),
            "parts": LLMPartsAdvisor(llm, agents_prompt, lang=self.lang),
            "maintenance": LLMMaintenanceAdvisor(llm, agents_prompt, lang=self.lang),
            "fault": LLMFaultHandler(llm, agents_prompt, lang=self.lang),
            "general": LLMGeneralAgent(llm, agents_prompt, lang=self.lang),
        }
        self.evaluator = LLMEvaluatorAgent(llm, self.prompts, lang=self.lang)
        self.aggregator = LLMAggregatorAgent(llm, self.prompts, lang=self.lang)
        self.orchestrator = Orchestrator(
            self.agents,
            llm_router=self.router,
            evaluator=self.evaluator,
            aggregator=self.aggregator,
        )
        self.safety_checker = None
        if bool(self.cfg.safety.get("use_llm", False)):
            self.safety_checker = LLMSafetyChecker(llm, agents_prompt, lang=self.lang)

    def _route(self, question: str, step_id: str, focus: str, detections: List[str]) -> str:
        previous_intent = self.memory.meta.get("last_route", "")
        previous_entities = ", ".join(self.memory.meta.get("last_entities", []))
        try:
            return self.router.classify(
                step_id=step_id,
                focus=focus,
                detections=", ".join(detections),
                question=question,
                previous_intent=previous_intent,
                previous_entities=previous_entities,
            )
        except Exception:
            return route(question)

    def _resolve_focus(self, route_id: str, question: str, det_names: List[str], focus_name: str) -> List[str]:
        if route_id not in {"assembly", "parts", "maintenance", "fault"} or not det_names:
            return det_names
        if self.entity_linker is None:
            return [focus_name] if focus_name else det_names
        try:
            picked = self.entity_linker.resolve_mention(question=question, visible=det_names, focus=focus_name)
        except Exception:
            picked = ""
        if picked:
            return [picked]
        return [focus_name] if focus_name else det_names

    def _run_topology(self, topology: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if topology == "shared":
            return self.orchestrator.run_shared(payload)
        if topology == "central":
            return self.orchestrator.run_central(payload)
        if topology == "hier":
            return self.orchestrator.run_hier(payload)
        if topology == "debate":
            return self.orchestrator.run_debate(payload, rounds=2)
        return self.orchestrator.run_mica(payload)

    def answer(self, question: str, decision: FrameDecision, topology: str = "") -> QAResponse:
        if not bool(self.cfg.ablation.get("use_mica_core", True)):
            return QAResponse(
                route="disabled",
                answer="MICA-core disabled by ablation config.",
                latency_ms=0,
                topology="disabled",
            )

        step_id = decision.display_step
        det_names_full = [str(item["name"]) for item in decision.perception.fused_detections]
        focus_name = str(decision.perception.relevant_detections[0].get("name", "")) if decision.perception.relevant_detections else ""
        route_id = self._route(question, step_id, focus_name, det_names_full)
        det_names = self._resolve_focus(route_id, question, det_names_full, focus_name)
        kb_snippets = build_kb_snippets_for_route(route_id, det_names, self.kb_index)

        if route_id == "parts" and kb_snippets.strip().lower() == "n/a":
            try:
                kb_snippets = parts_advisor.answer({"detections": det_names, "focus": focus_name, "focus_only": True}, self.kb)
            except Exception:
                kb_snippets = "n/a"

        payload = {
            "step_id": step_id,
            "focus": focus_name,
            "detections": ", ".join(det_names),
            "detections_full": ", ".join(det_names_full),
            "kb_snippets": kb_snippets,
            "history": self.memory.render_text(),
            "question": question,
            "previous_intent": self.memory.meta.get("last_route", ""),
            "previous_entities": ", ".join(self.memory.meta.get("last_entities", [])),
        }
        chosen_topology = topology or str(self.cfg.agent.get("topology", "mica"))
        t0 = time.perf_counter()
        output = self._run_topology(chosen_topology, payload)
        latency_ms = output.get("latency_ms")
        if latency_ms is None:
            latency_ms = int((time.perf_counter() - t0) * 1000)

        answer = output["final"]
        kb_safety = collect_kb_safety(det_names, self.kb_index)
        if bool(self.cfg.safety.get("rule_append", False)):
            answer = safety_check(answer, kb_safety)

        safety_audit = None
        if self.safety_checker is not None:
            safety_audit = self.safety_checker.audit(
                step_id=step_id,
                focus=focus_name,
                detections=", ".join(det_names),
                kb_safety=kb_safety,
                assistant_answer=answer,
            )
            append_on = {str(item).upper() for item in self.cfg.safety.get("append_on", ["UNSAFE"])}
            verdict = str(safety_audit.get("verdict", "")).upper()
            if verdict in append_on:
                reasons = safety_audit.get("reasons", []) or ["n/a"]
                fixes = safety_audit.get("fix", []) or ["n/a"]
                answer += (
                    f"\n\n[SAFETY AUDIT] Verdict: {verdict}\n"
                    f"Reasons:\n- " + "\n- ".join(reasons) + "\n"
                    f"Corrections:\n- " + "\n- ".join(fixes)
                )

        self.memory.append("user", question)
        self.memory.append("assistant", answer)
        self.memory.meta["last_route"] = route_id
        self.memory.meta["last_entities"] = det_names[:]

        return QAResponse(
            route=route_id,
            answer=answer,
            latency_ms=int(latency_ms),
            topology=chosen_topology,
            answers=output.get("answers", []),
            safety_audit=safety_audit,
        )
