"""Rule-based step predictor derived from KB workflow and detected parts.
   Compatible with:
   1) requires: {all_of, any_of, forbid}
   2) requires: {variants: [ {all_of, any_of, forbid}, ... ]}
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from collections import Counter

# ===== helpers =====
def _canon_counts(dets: List[Dict[str, Any]]) -> Counter:
    names = [str(d.get("name", "")).strip().lower() for d in dets]
    return Counter(names)

def _score_one_require(req: Dict[str, Any], cnt: Counter) -> Tuple[float, float, float, bool]:
    """Return (all_score, any_score, forbid_pen, forbid_hit) for a single require dict."""
    all_of = {str(k).strip().lower(): int(v)
              for k, v in (req.get("all_of", {}) or {}).items()}
    any_of = {str(k).strip().lower(): int(v)
              for k, v in (req.get("any_of", {}) or {}).items()}
    forbid = [str(x).strip().lower()
              for x in (req.get("forbid", []) or [])]

    # all_of coverage ∈ [0,1]
    cov = []
    for k, need in all_of.items():
        got = int(cnt.get(k, 0))
        cov.append(min(1.0, got / float(max(1, need))))
    all_score = sum(cov) / max(1, len(cov)) if cov else 1.0

    # any_of satisfied?
    any_score = 1.0 if (not any_of or any(cnt.get(k, 0) >= v for k, v in any_of.items())) else 0.0

    # forbid
    bad = any(cnt.get(x, 0) > 0 for x in forbid)
    forbid_pen = 0.5 if bad else 0.0

    return all_score, any_score, forbid_pen, bad

def _best_conf_for_requires(req: Dict[str, Any], cnt: Counter) -> Tuple[float, Dict[str, float]]:
    """Handle either single require or variants; return best conf and its components."""
    variants = req.get("variants", None)
    if variants and isinstance(variants, list):
        best, expl = -1.0, {}
        for r in variants:
            a, y, fp, _ = _score_one_require(r, cnt)
            conf = max(0.0, min(1.0, 0.6 * a + 0.4 * y - fp))
            if conf > best:
                best = conf
                expl = {"all_score": a, "any_score": y, "forbid_pen": fp}
        return (best if best >= 0 else 0.0), expl
    # single require
    a, y, fp, _ = _score_one_require(req, cnt)
    conf = max(0.0, min(1.0, 0.6 * a + 0.4 * y - fp))
    return conf, {"all_score": a, "any_score": y, "forbid_pen": fp}

# ===== public API =====
def predict_by_rules(kb: Dict[str, Any], dets: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, Any]]:
    """
    State-graph step prediction from KB workflow.
    Supports:
      - requires: {all_of, any_of, forbid}
      - requires: {variants: [ {all_of, any_of, forbid}, ... ]}
    Returns:
      (step_id, confidence, explanation)
    """
    wf = kb.get("workflow", []) or []
    if not wf:
        return "S1", 0.4, {"reason": "no_workflow"}

    cnt = _canon_counts(dets)
    best_sid, best_conf, best_expl = "S1", -1.0, {}

    for s in wf:
        sid = str(s.get("id", "S1")).strip().upper()
        req = s.get("requires", {}) or {}
        conf, expl = _best_conf_for_requires(req, cnt)
        if conf > best_conf:
            best_sid, best_conf, best_expl = sid, conf, expl

    return best_sid, float(best_conf), best_expl

def compat_vector(kb: Dict[str, Any], dets: List[Dict[str, Any]], steps: List[str]) -> List[float]:
    """
    Per-step compatibility in [0,1] for the given 'steps' list.
    If a step has variants, take the max compatibility among variants.
    """
    wf = kb.get("workflow", []) or []
    cnt = _canon_counts(dets)
    req_by_id = {str(s.get("id", "")).strip().upper(): s.get("requires", {}) for s in wf}

    vec: List[float] = []
    for sid in steps:
        req = req_by_id.get(str(sid).strip().upper(), {})
        variants = req.get("variants", None)
        if variants and isinstance(variants, list):
            best = 0.0
            for r in variants:
                a, y, fp, bad = _score_one_require(r, cnt)
                comp = 0.0 if bad else (0.6 * a + 0.4 * y)
                if comp > best:
                    best = comp
            vec.append(float(max(0.0, min(1.0, best))))
        else:
            a, y, fp, bad = _score_one_require(req, cnt)
            comp = 0.0 if bad else (0.6 * a + 0.4 * y)
            vec.append(float(max(0.0, min(1.0, comp))))
    return vec
