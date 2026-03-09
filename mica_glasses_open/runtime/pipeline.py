"""Shared pipeline orchestration for offline and live runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from mica_glasses_open.legacy_impl.core.asf import ASF
from mica_glasses_open.legacy_impl.core.context_extraction import select_by_depth
from mica_glasses_open.legacy_impl.core.depth import DepthStub
from mica_glasses_open.legacy_impl.core.entity_linker import LLMEntityLinker
from mica_glasses_open.legacy_impl.core.fusion import fuse_window
from mica_glasses_open.legacy_impl.core.kb import build_alias_map
from mica_glasses_open.legacy_impl.core.llm import LLM
from mica_glasses_open.legacy_impl.core.retrieval import GalleryIndex
from mica_glasses_open.legacy_impl.core.yolo import YOLODetector
from mica_glasses_open.paper_modules.adaptive_assembly_step_recognition import (
    AdaptiveAssemblyStepRecognition,
)
from mica_glasses_open.paper_modules.depth_guided_object_context_extraction import (
    DepthGuidedObjectContextExtraction,
)
from mica_glasses_open.paper_modules.mica_core import MICACore, load_prompts
from mica_glasses_open.types import FrameDecision


class SlidingWindowFuser:
    """Maintain the detection window used by Eq. 7 style fusion."""

    def __init__(self, window: int, iou_thr: float) -> None:
        self.window = int(window)
        self.iou_thr = float(iou_thr)
        self._raw: list[list[dict[str, Any]]] = []

    def update(self, raw_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._raw.append(raw_detections)
        if len(self._raw) > self.window:
            self._raw.pop(0)
        return fuse_window(list(self._raw), iou_thr=self.iou_thr) if self._raw else raw_detections


class ContextSelector:
    """Thin wrapper around the original depth-guided selector."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def select(self, detections, depth_map):
        return select_by_depth(
            detections,
            depth_map,
            tau_p=float(self.cfg.depth_context.get("tau_p", 80.0)),
            tau_d=float(self.cfg.depth_context.get("tau_d", 0.12)),
        )


class MICAPipeline:
    """Own the three paper modules plus runtime state."""

    def __init__(self, cfg, kb: Dict[str, Any], yolo_weights: str, run_dir: Path, device: str = "cpu") -> None:
        self.cfg = cfg
        self.kb = kb
        self.run_dir = Path(run_dir)
        self.device = device

        llm_cfg = self.cfg.llm
        self.prompts = load_prompts()
        self.llm = LLM(
            model=str(llm_cfg.get("model", "qwen2.5:7b-instruct")),
            url=str(llm_cfg.get("url", "http://localhost:11434/api/generate")),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            max_tokens=int(llm_cfg.get("max_tokens", 160)),
            timeout=int(llm_cfg.get("timeout", 30)),
        )
        self.entity_linker = LLMEntityLinker(
            self.llm,
            self.prompts,
            lang=self.prompts.get("lang", "en"),
            cache_path=self.run_dir / "entity_linker_cache.json",
        )

        detector = YOLODetector(
            yolo_weights,
            device=device,
            nms_iou=float(self.cfg.detection.get("nms_iou", 0.5)),
            use_builtin_tta=bool(self.cfg.detection.get("use_builtin_tta", True)),
        )
        depth_model = DepthStub(device=device)
        temporal_fuser = SlidingWindowFuser(
            window=int(self.cfg.fusion.get("window", 5)),
            iou_thr=float(self.cfg.fusion.get("iou_thr", 0.5)),
        )
        self.object_context_module = DepthGuidedObjectContextExtraction(
            cfg=self.cfg,
            kb=self.kb,
            alias_map=build_alias_map(self.kb),
            detector=detector,
            depth_model=depth_model,
            temporal_fuser=temporal_fuser,
            context_selector=ContextSelector(self.cfg),
            entity_linker=self.entity_linker,
        )
        gallery = GalleryIndex(
            root=str(self.cfg.gallery.get("root", "")),
            exts=list(self.cfg.gallery.get("exts", [".jpg", ".jpeg", ".png", ".bmp"])),
            embed_mode=str(self.cfg.gallery.get("embed", "rgb-mean-8x8")),
        )
        workflow = self.kb.get("workflow", []) or []
        steps_cfg = self.cfg.asf.get("steps") or [
            str(item.get("id", "")).strip().upper() for item in workflow if item.get("id")
        ] or ["S1", "S2", "S3", "S4"]
        asf_head = ASF(
            w_init=self.cfg.asf.get("w_init", {"s": 0.5, "r": 0.5}),
            beta=float(self.cfg.asf.get("beta", 0.9)),
            persist=self.run_dir / "asf_weights.json",
            K=int(self.cfg.asf.get("K", len(steps_cfg))),
            steps=list(steps_cfg),
            leak=self.cfg.asf.get("leak", {"s": 0.05, "r": 0.05}),
            clamp=self.cfg.asf.get("clamp", {"lo": 0.05, "hi": 0.95}),
            reward=float(self.cfg.asf.get("reward", 1.05)),
            penalty=float(self.cfg.asf.get("penalty", 0.90)),
            conf_freeze=float(self.cfg.asf.get("conf_freeze", 0.80)),
            gamma=float(self.cfg.asf.get("gamma", 2.0)),
            eta=float(self.cfg.asf.get("eta", 0.10)),
            trust=float(self.cfg.asf.get("trust", 0.15)),
            transitions=self.cfg.asf.get("transitions", {}) or {},
            lambda_trans=float(self.cfg.asf.get("lambda_trans", 2.0)),
            lambda_rule=float(self.cfg.asf.get("lambda_rule", 1.0)),
            expo_rho=float(self.cfg.asf.get("expo_rho", 0.5)),
            b_cap=float(self.cfg.asf.get("b_cap", 0.5)),
            floor_per_class=float(self.cfg.asf.get("floor_per_class", 0.02)),
            balance_window=int(self.cfg.asf.get("balance_window", 50)),
            balance_tau=float(self.cfg.asf.get("balance_tau", 0.6)),
        )
        self.step_module = AdaptiveAssemblyStepRecognition(
            cfg=self.cfg,
            kb=self.kb,
            gallery=gallery,
            asf_head=asf_head,
            run_dir=self.run_dir,
        )
        self.mica_core = MICACore(self.cfg, self.kb, self.llm, entity_linker=self.entity_linker)

        self.proc_iter = 0
        self.prev_signature = ()
        self.stable_count = 0
        self.current_step = ""
        self.current_conf = 0.0
        self.last_fused_step = ""

    def process_frame(self, frame_bgr, frame_index: int) -> FrameDecision:
        perception = self.object_context_module.process(frame_bgr)
        prediction = self.step_module.predict(frame_bgr, perception, prev_step=self.last_fused_step or None)

        if perception.signature and perception.signature == self.prev_signature:
            self.stable_count += 1
        else:
            self.stable_count = 1 if perception.signature else 0
        self.prev_signature = perception.signature
        is_stable = self.stable_count >= int(self.cfg.stability.get("stable_n", 5))

        if is_stable:
            self.current_step = prediction.fused_step
            self.current_conf = prediction.fused_conf

        display_step = self.current_step or prediction.fused_step
        display_conf = self.current_conf or prediction.fused_conf
        self.last_fused_step = prediction.fused_step

        decision = FrameDecision(
            iter_index=self.proc_iter,
            frame_index=frame_index,
            is_stable=bool(is_stable),
            stable_count=self.stable_count,
            display_step=display_step,
            display_conf=float(display_conf),
            perception=perception,
            step_prediction=prediction,
        )
        self.proc_iter += 1
        return decision

    def apply_feedback(self, user_step: str, decision: FrameDecision) -> None:
        self.step_module.apply_feedback(user_step, decision.step_prediction)
        self.current_step = str(user_step).strip().upper()
        self.current_conf = max(float(decision.display_conf), float(decision.step_prediction.fused_conf))
        self.last_fused_step = self.current_step

    def answer_question(self, question: str, decision: FrameDecision, topology: str = ""):
        chosen_topology = topology or str(self.cfg.agent.get("topology", "mica"))
        return self.mica_core.answer(question, decision, topology=chosen_topology)
