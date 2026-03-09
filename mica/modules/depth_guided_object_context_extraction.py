"""Paper module: Depth-guided Object Context Extraction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mica.runtime.kb_utils import kb_components_list, kb_index_by_name
from mica.types import PerceptionContext


Detection = Dict[str, Any]


def signature(detections: List[Detection]) -> Tuple[Tuple[str, int], ...]:
    names = [str(item.get("name", "")).lower() for item in detections]
    return tuple(sorted(Counter(names).items()))


def canonicalize_names(detections: List[Detection], alias_map: Dict[str, str]) -> List[Detection]:
    output: List[Detection] = []
    for item in detections:
        name = str(item.get("name", "")).strip().lower().replace("_", " ")
        canonical = alias_map.get(name, name)
        updated = dict(item)
        updated["name"] = canonical
        output.append(updated)
    return output


def draw_overlay(frame_bgr: np.ndarray, detections: List[Detection], step_text: str) -> np.ndarray:
    canvas = frame_bgr.copy()
    for item in detections:
        x1, y1, x2, y2 = [int(value) for value in item["xyxy"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{item['name']} {item.get('conf', 0.0):.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        step_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (50, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def safe_crop(frame_bgr: np.ndarray, detections: List[Detection], pad: int = 8) -> Optional[np.ndarray]:
    if not detections:
        return None
    x1, y1, x2, y2 = [int(value) for value in detections[0]["xyxy"]]
    height, width = frame_bgr.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width - 1, x2 + pad)
    y2 = min(height - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


class DepthGuidedObjectContextExtraction:
    """Compose detection, temporal fusion, depth estimation, and context selection."""

    def __init__(
        self,
        cfg,
        kb: Dict[str, Any],
        alias_map: Dict[str, str],
        detector,
        depth_model,
        temporal_fuser,
        context_selector,
        entity_linker=None,
    ) -> None:
        self.cfg = cfg
        self.kb = kb
        self.alias_map = alias_map
        self.detector = detector
        self.depth_model = depth_model
        self.temporal_fuser = temporal_fuser
        self.context_selector = context_selector
        self.entity_linker = entity_linker
        self.components = kb_components_list(kb)
        self.kb_index = kb_index_by_name(self.components)

    def _canonicalize(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return detections
        observed = [str(item.get("name", "")) for item in detections]
        canonical_pool = [str(component.get("name", "")).strip() for component in self.components if component.get("name")]
        if not self.entity_linker or not observed or not canonical_pool:
            return canonicalize_names(detections, self.alias_map)
        try:
            canon_map = self.entity_linker.canonicalize(observed, canonical_pool)
        except Exception:
            return canonicalize_names(detections, self.alias_map)
        if len(set(canon_map.values())) <= 1 and len(set(observed)) > 1:
            return canonicalize_names(detections, self.alias_map)
        output: List[Detection] = []
        for item in detections:
            updated = dict(item)
            updated["name"] = canon_map.get(item["name"], item["name"])
            output.append(updated)
        return output

    def process(self, frame_bgr: np.ndarray) -> PerceptionContext:
        raw_detections = self.detector.detect(
            frame_bgr,
            conf=float(self.cfg.detection.get("conf", 0.25)),
            tta=bool(self.cfg.detection.get("tta", True)),
        )
        fused_detections = self.temporal_fuser.update(raw_detections)
        fused_detections = self._canonicalize(fused_detections)
        depth_map = self.depth_model.infer(frame_bgr)
        relevant_detections, nearest_index = self.context_selector.select(fused_detections, depth_map)
        if not bool(self.cfg.ablation.get("use_depth_guided_context", True)):
            relevant_detections = list(fused_detections)
        return PerceptionContext(
            raw_detections=raw_detections,
            fused_detections=fused_detections,
            relevant_detections=relevant_detections,
            nearest_index=nearest_index,
            depth_map=depth_map,
            signature=signature(fused_detections),
        )
