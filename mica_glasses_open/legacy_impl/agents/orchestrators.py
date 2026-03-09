from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time
import re

def _measure(fn, *a, **kw) -> Tuple[str,int]:
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ms  = int((time.perf_counter()-t0)*1000)
    return out, ms

def _sanitize(s: str) -> str:
    if not s: return s
    for tok in ("<<SYS>>","<<USR>>","<|system|>","<|user|>","SYSTEM:","USER:","[Agent]","[central]"):
        s = s.replace(tok, "")
    return s.strip()

def _shorten(s: str, limit: int = 600) -> str:
    s = s or ""
    return s if len(s) <= limit else (s[:limit] + " ...")



class Orchestrator:
    def __init__(self, agents: Dict[str, Any], llm_router=None,
                 evaluator=None, aggregator=None):
        self.agents = agents
        self.router = llm_router
        self.evaluator = evaluator     # LLMEvaluatorAgent
        self.aggregator = aggregator   # LLMAggregatorAgent

    def run_mica(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        MICA topology: use the LLM router to pick ONE best agent,
        then call that agent only. Returns a dict with route, answers list, final text and latency.
        """
        # Prepare routing inputs from payload
        step_id = payload.get("step_id", "")
        focus = payload.get("focus", "")
        # prefer full detections (for better routing on count/general), fallback to detections
        detections = payload.get("detections_full", payload.get("detections", ""))
        question = payload.get("question", "")
        prev_int = payload.get("previous_intent", "")
        prev_ents = payload.get("previous_entities", "")

        # Route
        if self.router is not None:
            try:
                route_id = self.router.classify(
                    step_id=step_id, focus=focus, detections=detections,
                    question=question, previous_intent=prev_int, previous_entities=prev_ents
                )
            except Exception:
                # tiny heuristic fallback
                route_id = "general"
        else:
            route_id = "general"

        # Dispatch
        agent = self.agents.get(route_id)
        if agent is None:
            # final safety net
            agent = self.agents.get("general")

        text, ms = _measure(agent.answer, **payload)
        return {
            "route": route_id,
            "answers": [(route_id, text, ms)],  # keep a consistent format with other topologies
            "final": text,
            "latency_ms": ms
        }
    def run_shared(self, payload: Dict[str,Any]) -> Dict[str, Any]:
        answers = []
        lat = 0
        for k,agent in self.agents.items():
            text, ms = _measure(agent.answer, **payload)
            lat += ms
            answers.append((k,text,ms))
        # LLM evaluator if available
        if self.evaluator:
            data = self.evaluator.select(
                question=payload.get("question",""),
                step_id=payload.get("step_id",""),
                focus=payload.get("focus",""),
                detections=payload.get("detections_full", payload.get("detections","")),
                candidates=[t for _,t,_ in answers]
            )
            idx = int(data.get("best_index", 0))
            idx = max(0, min(idx, len(answers)-1))
            final = answers[idx][1]
            lat += 0  # evaluator latency已包含在 LLM.generate 内部不可见；如需记录可在 evaluator 内返回
            return {"route":"shared","answers":answers,"final":final,"latency_ms":lat}
        # fallback heuristic
        best = max(answers, key=lambda x: (len(x[1])>0, -len(x[1])))
        return {"route":"shared","answers":answers,"final":best[1],"latency_ms":lat}

    def run_central(self, payload: Dict[str,Any]) -> Dict[str, Any]:
        answers=[]
        lat=0

        # 1) 广播：收集所有子 agent 的候选 (清洗 + 限长)
        for k,agent in self.agents.items():
            text, ms = _measure(agent.answer, **payload)
            lat+=ms
            answers.append((k,_sanitize(text),ms))

        # 2) 聚合
        merged = ""
        if self.aggregator:
            cand_texts = [t for _, t, _ in answers if t]
            if cand_texts:
                t0 = time.perf_counter()
                merged = self.aggregator.merge(
                    question=payload.get("question", ""),
                    step_id=payload.get("step_id", ""),
                    focus=payload.get("focus", ""),
                    detections=payload.get("detections_full", payload.get("detections", "")),
                    answers=cand_texts
                )
                lat += int((time.perf_counter() - t0) * 1000)

        final = _sanitize(merged)

        def _is_bad(txt: str) -> bool:
            if not txt or len(txt) < 3: return True
            low = txt.lower()
            bad_cues = [
                "not sure what you mean", "unclear", "not enough information",
                "内容似乎不完整", "我不太明白", "请提供更多", "请澄清"
            ]
            return any(c in low for c in bad_cues)

        if _is_bad(final):
            # 回退策略：优先 parts/maintenance/assembly 的首行；否则取最短的非空候选
            priority = ["parts", "maintenance", "assembly", "fault", "general"]
            bykey = {k: t for k, t, _ in answers}
            for k in priority:
                if bykey.get(k):
                    final = bykey[k]
                    break
            if _is_bad(final):
                non_empty = [t for _, t, _ in answers if t.strip()]
                final = min(non_empty, key=len) if non_empty else ""

        return {"route": "central", "answers": answers, "final": final, "latency_ms": lat}

    def run_hier(self, payload: Dict[str,Any]) -> Dict[str, Any]:
        """Hierarchical relay: Maintenance -> Assembly -> Parts -> Fault -> General."""
        order=["maintenance","assembly","parts","fault","general"]
        answers=[]
        cur_payload=dict(payload)
        for k in order:
            agent=self.agents[k]
            text, ms = _measure(agent.answer, **cur_payload)
            answers.append((k,text,ms))
            # relay: append each output to history for the next agent
            hist = cur_payload.get("history","")
            cur_payload["history"] = (hist+"\nASSISTANT("+k+"):"+text).strip()
        final = answers[-1][1] if answers else ""
        return {"route":"hier","answers":answers,"final":final,"latency_ms":sum(ms for _,_,ms in answers)}

    def run_debate(self, payload: Dict[str,Any], rounds:int=2) -> Dict[str, Any]:
        """Peer debate + voting. Each round agents critique others briefly."""
        # init: each proposes
        answers=[]
        props={}
        lat=0
        for k,agent in self.agents.items():
            text, ms = _measure(agent.answer, **payload)
            lat += ms
            props[k]=text
            answers.append((k,text,ms))
        # debate rounds (short)
        for _ in range(max(1,rounds-1)):
            new_props={}
            for k,agent in self.agents.items():
                # critic prompt injected by history
                hist = payload.get("history","")
                critique = "\n".join([f"{kk}:{vv}" for kk,vv in props.items() if kk!=k])
                pl = dict(payload)
                pl["history"] = (hist + f"\nPEER RESPONSES:\n{critique}\nPlease refine your answer.").strip()
                text, ms = _measure(agent.answer, **pl)
                lat+=ms
                new_props[k]= text
            props = new_props
        # voting: score by heuristic token coverage + brevity
        scored = []
        for k,v in props.items():
            s = (("safety" in v.lower()) + ("tool" in v.lower()) + ("step" in v.lower())) - 0.002*len(v)
            scored.append((s,k,v))
        scored.sort(reverse=True)
        final = scored[0][2] if scored else ""
        return {"route":"debate","answers":answers+[(k,props[k],0) for k in props], "final":final, "latency_ms":lat}
