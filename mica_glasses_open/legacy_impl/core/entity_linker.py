"""LLM-first entity linker: canonicalization and mention resolution, with caching."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from mica_glasses_open.legacy_impl.core.llm import LLM
from mica_glasses_open.legacy_impl.core.llm_json import ask_json

class LLMEntityLinker:
    """
    LLM-based canonicalizer and resolver with simple on-disk caching.
    """

    def __init__(self, llm: LLM, prompts: Dict[str, Any], lang: str = "en",
                 cache_path: Optional[Path] = None):
        self.llm = llm
        self.p_canon = prompts.get("entity_linker", {}).get("canon", {})
        self.p_resolve = prompts.get("entity_linker", {}).get("resolve", {})
        self.lang = lang
        self.cache_path = cache_path
        self.cache = {"canon": {}, "resolve": {}}
        if cache_path and cache_path.exists():
            try:
                self.cache.update(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    def _save(self):
        if self.cache_path:
            try:
                self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
            except Exception:
                pass

    def canonicalize(self, observed: List[str], canonical: List[str]) -> Dict[str, str]:
        """
        Map observed labels to canonical names (LLM picks from 'canonical' list).
        """
        # caching key
        key = json.dumps({"obs": observed, "can": canonical}, sort_keys=True)
        if key in self.cache["canon"]:
            return self.cache["canon"][key]

        sys_p = self.p_canon.get("system", "")
        usr_p = self.p_canon.get("user", "").format(
            canonical_list="\n".join(f"- {c}" for c in canonical),
            observed_list="\n".join(f"- {o}" for o in observed)
        )
        data = ask_json(self.llm, sys_p, usr_p, fallback={"map": {o: canonical[0] if canonical else o for o in observed}})
        mapping = data.get("map", {}) if isinstance(data, dict) else {}
        # minimal post-check: ensure outputs are in canonical list
        can_set = set(canonical)
        result = {}
        for o in observed:
            c = mapping.get(o, canonical[0] if canonical else o)
            if c not in can_set and canonical:
                c = canonical[0]
            result[o] = c
        self.cache["canon"][key] = result
        self._save()
        return result

    def resolve_mention(self, question: str, visible: List[str], focus: str = "") -> str:
        """
        Pick which visible name the question refers to. If empty, caller decides (e.g., use focus).
        """
        key = json.dumps({"q": question, "vis": visible, "f": focus}, sort_keys=True)
        if key in self.cache["resolve"]:
            return self.cache["resolve"][key]

        sys_p = self.p_resolve.get("system", "")
        usr_p = self.p_resolve.get("user", "").format(
            question=question,
            visible_list=", ".join(visible),
            focus=focus or ""
        )
        data = ask_json(self.llm, sys_p, usr_p, fallback={"pick": focus or (visible[0] if visible else "")})
        pick = data.get("pick", "") if isinstance(data, dict) else ""
        if pick and pick not in visible and visible:
            pick = visible[0]
        self.cache["resolve"][key] = pick
        self._save()
        return pick
