"""Paper module: Adaptive Assembly Step Recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mica_glasses_open.types import PerceptionContext, StepPredictionBundle


class AdaptiveAssemblyStepRecognition:
    """Compose the state-graph expert, retrieval expert, and ASF head."""

    def __init__(self, cfg, kb: Dict[str, Any], gallery, asf_head, run_dir: Path) -> None:
        self.cfg = cfg
        self.kb = kb
        self.gallery = gallery
        self.asf_head = asf_head
        self.run_dir = Path(run_dir)
        self.steps = self._resolve_steps()
        self.gallery_error = ""
        self.gallery_size = self._build_gallery()

    def _resolve_steps(self) -> List[str]:
        workflow = self.kb.get("workflow", []) or []
        steps_from_kb = [str(item.get("id", "")).strip().upper() for item in workflow if item.get("id")]
        configured = self.cfg.asf.get("steps") or steps_from_kb or ["S1", "S2", "S3", "S4"]
        return [str(step).strip().upper() for step in configured]

    def _build_gallery(self) -> int:
        gallery_root = str(self.cfg.gallery.get("root", "")).strip()
        if not gallery_root:
            self.gallery_error = "gallery_root_not_set"
            return 0
        try:
            return int(self.gallery.build())
        except Exception as exc:
            self.gallery_error = str(exc)
            return 0

    def _state_prediction(self, detections: List[Dict[str, Any]]):
        from mica_glasses_open.legacy_impl.core.step_rules import compat_vector, predict_by_rules

        step_id, confidence, meta = predict_by_rules(self.kb, detections)
        compat = compat_vector(self.kb, detections, self.steps)
        return step_id, float(confidence), meta, compat

    def _retrieval_prediction(self, frame_bgr):
        if not bool(self.cfg.ablation.get("use_retrieval_expert", True)):
            return self.steps[0], 0.0, {"reason": "retrieval_disabled"}
        if self.gallery.items:
            return self.gallery.predict(frame_bgr, topk=int(self.cfg.gallery.get("topk", 5)))
        return self.steps[0], 0.0, {"reason": self.gallery_error or "empty_gallery"}

    def predict(
        self,
        frame_bgr,
        perception: PerceptionContext,
        prev_step: Optional[str] = None,
    ) -> StepPredictionBundle:
        use_state = bool(self.cfg.ablation.get("use_state_graph_expert", True))
        use_retrieval = bool(self.cfg.ablation.get("use_retrieval_expert", True))
        use_asf = bool(self.cfg.ablation.get("use_asf", True))

        state_input = perception.fused_detections
        if bool(self.cfg.depth_context.get("use_relevant_for_rules", False)) and perception.relevant_detections:
            state_input = perception.relevant_detections

        if use_state:
            state_step, state_conf, state_meta, compat = self._state_prediction(state_input)
        else:
            state_step = self.steps[0]
            state_conf = 0.0
            state_meta = {"reason": "state_graph_disabled"}
            compat = [0.0] * len(self.steps)

        retrieval_step, retrieval_conf, retrieval_meta = self._retrieval_prediction(frame_bgr)
        if not use_retrieval:
            retrieval_step = self.steps[0]
            retrieval_conf = 0.0
            retrieval_meta = {"reason": "retrieval_disabled"}

        if use_asf and use_state and use_retrieval:
            fused_step, fused_conf, fusion_meta = self.asf_head.fuse(
                (state_step, state_conf),
                (retrieval_step, retrieval_conf),
                prev_step=prev_step,
                compat=compat,
            )
            if fused_step == state_step and fused_step != retrieval_step:
                chosen = "s"
            elif fused_step == retrieval_step and fused_step != state_step:
                chosen = "r"
            else:
                chosen = "s" if state_conf >= retrieval_conf else "r"
            weights = {
                "s": float(fusion_meta.get("g_s", self.asf_head.g[0])),
                "r": float(fusion_meta.get("g_r", self.asf_head.g[1])),
            }
            fusion_meta = dict(fusion_meta)
            fusion_meta["chosen"] = chosen
        else:
            candidates = []
            if use_state:
                candidates.append(("s", state_step, state_conf))
            if use_retrieval:
                candidates.append(("r", retrieval_step, retrieval_conf))
            if candidates:
                chosen, fused_step, fused_conf = max(candidates, key=lambda item: float(item[2]))
            else:
                chosen, fused_step, fused_conf = "none", self.steps[0], 0.0
            weights = {
                "s": 1.0 if chosen == "s" else 0.0,
                "r": 1.0 if chosen == "r" else 0.0,
            }
            fusion_meta = {"reason": "asf_disabled_or_single_expert", "chosen": chosen}

        return StepPredictionBundle(
            state_step=state_step,
            state_conf=float(state_conf),
            retrieval_step=retrieval_step,
            retrieval_conf=float(retrieval_conf),
            fused_step=fused_step,
            fused_conf=float(fused_conf),
            chosen=chosen,
            weights=weights,
            compat=compat,
            state_meta=state_meta,
            retrieval_meta=retrieval_meta,
            fusion_meta=fusion_meta,
        )

    def apply_feedback(self, user_step: str, prediction: StepPredictionBundle) -> None:
        if not (
            bool(self.cfg.ablation.get("use_asf", True))
            and bool(self.cfg.ablation.get("use_state_graph_expert", True))
            and bool(self.cfg.ablation.get("use_retrieval_expert", True))
        ):
            return
        self.asf_head.update_with_feedback_plus(
            user_step=user_step,
            s_pred=prediction.state_step,
            r_pred=prediction.retrieval_step,
            chosen=prediction.chosen,
            fused=prediction.fused_step,
            s_conf=prediction.state_conf,
            r_conf=prediction.retrieval_conf,
        )
