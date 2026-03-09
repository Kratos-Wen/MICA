"""Shared dataclasses for the modular MICA runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Detection = Dict[str, Any]


@dataclass
class PerceptionContext:
    raw_detections: List[Detection]
    fused_detections: List[Detection]
    relevant_detections: List[Detection]
    nearest_index: Optional[int]
    depth_map: Any
    signature: Tuple[Tuple[str, int], ...]


@dataclass
class StepPredictionBundle:
    state_step: str
    state_conf: float
    retrieval_step: str
    retrieval_conf: float
    fused_step: str
    fused_conf: float
    chosen: str
    weights: Dict[str, float] = field(default_factory=dict)
    compat: List[float] = field(default_factory=list)
    state_meta: Dict[str, Any] = field(default_factory=dict)
    retrieval_meta: Dict[str, Any] = field(default_factory=dict)
    fusion_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResponse:
    route: str
    answer: str
    latency_ms: int
    topology: str
    answers: List[Tuple[str, str, int]] = field(default_factory=list)
    safety_audit: Optional[Dict[str, Any]] = None


@dataclass
class FrameDecision:
    iter_index: int
    frame_index: int
    is_stable: bool
    stable_count: int
    display_step: str
    display_conf: float
    perception: PerceptionContext
    step_prediction: StepPredictionBundle
