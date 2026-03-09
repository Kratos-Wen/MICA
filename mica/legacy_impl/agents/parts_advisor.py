
"""Parts Advisor agent: describe visible parts using KB fields."""
from __future__ import annotations
from typing import Dict, Any, List
from mica.legacy_impl.core.kb import rag_fields_for_component

# def answer(context: Dict[str,Any], kb: Dict[str,Any]) -> str:
#     """
#     Summarize visible components (features, tools, safety).
#
#     Args:
#         context: Dict with 'detections' list of canonical names.
#         kb: Knowledge base.
#
#     Returns:
#         Textual description.
#     """
#     present = set(context.get("detections", []))
#     lines=[]
#     for comp in kb.get("components", []):
#         cname = str(comp.get("name","")).strip().lower()
#         if cname in present:
#             f = rag_fields_for_component(comp)
#             lines.append(f"* {comp.get('name')}: features={', '.join(f['key_features']) or 'n/a'}; tools={', '.join(f['tools']) or 'n/a'}; safety={', '.join(f['safety']) or 'n/a'}")
#     if not lines: return "No known components recognized in the current view."
#     return "Visible components:\n" + "\n".join(lines)

def answer(context, kb):
    present = set(context.get("detections", []))
    focus = context.get("focus")
    focus_only = context.get("focus_only", False)

    def describe(c):
        f = rag_fields_for_component(c)
        return f"* {c.get('name')}: features={', '.join(f['key_features']) or 'n/a'}; tools={', '.join(f['tools']) or 'n/a'}; safety={', '.join(f['safety']) or 'n/a'}"

    comps = {str(c.get("name","")).strip().lower(): c for c in kb.get("components", [])}

    # 单对象优先：如果有 focus 且在 KB 命中
    if focus_only and focus in comps:
        return "Part detail:\n" + describe(comps[focus])

    # 其次：汇总所有命中的部件
    lines=[]
    for name in present:
        if name in comps:
            lines.append(describe(comps[name]))
    if lines:
        return "Visible components:\n" + "\n".join(lines)

    # 最后回退：没有命中 KB，也给出检测名，避免“一个有、一个没有”的体验落差
    fallback = ", ".join(sorted(present)) or "none"
    return f"No KB entries matched. Detected: {fallback}"
