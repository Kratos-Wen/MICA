"""Offline video runner for the modular MICA package."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from mica.legacy_impl.core.kb import load_kb
from mica.paper_modules.depth_guided_object_context_extraction import draw_overlay
from mica.runtime.artifacts import RunArtifacts
from mica.runtime.interaction import handle_stable_console_interaction
from mica.runtime.kb_utils import ensure_parent
from mica.runtime.pipeline import MICAPipeline


def run_video(
    video_path: str,
    yolo_weights: str,
    kb_path: str,
    cfg,
    device: str = "cpu",
    interactive: bool = False,
    topology: str = "",
) -> Path:
    kb = load_kb(kb_path)
    artifacts = RunArtifacts.create(
        cfg,
        source_label=video_path,
        meta={
            "source": video_path,
            "mode": "video",
            "yolo_weights": yolo_weights,
            "kb_path": kb_path,
            "device": device,
        },
    )
    pipeline = MICAPipeline(cfg, kb, yolo_weights=yolo_weights, run_dir=artifacts.run_dir, device=device)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    writer = None
    output_fps = cfg.video.get("output_fps")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if output_fps is None:
        output_fps = source_fps
    if bool(cfg.video.get("write_annotated", False)):
        out_path = artifacts.build_annotated_video_path(Path(video_path).stem)
        ensure_parent(out_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(output_fps),
            (width, height),
        )

    frame_index = 0
    stride = int(cfg.video.get("stride", 3))
    interaction_enabled = bool(interactive and sys.stdin.isatty())

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % stride != 0:
                continue

            decision = pipeline.process_frame(frame_bgr, frame_index=frame_index)
            step_text = f"STEP: {decision.display_step} ({decision.display_conf:.2f})"
            canvas = draw_overlay(frame_bgr, decision.perception.fused_detections, step_text)

            if writer is not None:
                writer.write(canvas)
            artifacts.save_snapshot(canvas, decision.iter_index, frame_index)
            artifacts.log_iteration(decision)

            if interaction_enabled and decision.is_stable:
                handle_stable_console_interaction(pipeline, decision, artifacts, topology=topology)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        artifacts.finalize()

    return artifacts.run_dir
