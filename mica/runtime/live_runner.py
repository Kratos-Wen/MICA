"""Live camera runner for the modular MICA package."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from mica.legacy_impl.core.kb import load_kb
from mica.paper_modules.depth_guided_object_context_extraction import (
    draw_overlay,
    safe_crop,
)
from mica.runtime.artifacts import RunArtifacts
from mica.runtime.interaction import handle_stable_console_interaction
from mica.runtime.kb_utils import ensure_parent
from mica.runtime.pipeline import MICAPipeline
from mica.runtime.sources import open_camera_capture
from mica.runtime.ui import LiveUI


def run_camera(
    camera_index: int,
    yolo_weights: str,
    kb_path: str,
    cfg,
    device: str = "cpu",
    interactive: bool = False,
    topology: str = "",
) -> Path:
    kb = load_kb(kb_path)
    source_label = f"camera://{int(camera_index)}"
    artifacts = RunArtifacts.create(
        cfg,
        source_label=source_label,
        meta={
            "source": source_label,
            "mode": "camera",
            "yolo_weights": yolo_weights,
            "kb_path": kb_path,
            "device": device,
        },
    )
    pipeline = MICAPipeline(cfg, kb, yolo_weights=yolo_weights, run_dir=artifacts.run_dir, device=device)
    capture, backend_name = open_camera_capture(int(camera_index), cfg.camera)

    writer = None
    if bool(cfg.video.get("write_annotated", False)):
        out_path = artifacts.build_annotated_video_path(f"camera_{int(camera_index)}")
        ensure_parent(out_path)
        width = int(cfg.camera.get("width", 1280))
        height = int(cfg.camera.get("height", 720))
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(cfg.camera.get("fps", 30.0)),
            (width, height),
        )

    ui = LiveUI(
        enabled=True,
        window_name=str(cfg.camera.get("window_name", "MICA Live")),
        focus_window_name=str(cfg.camera.get("focus_window_name", "MICA Focus")),
        show_focus=bool(cfg.camera.get("show_focus", True)),
        show_help=True,
    )

    frame_index = 0
    stride = int(cfg.video.get("stride", 3))
    paused = False
    pending_prompt = bool(cfg.interaction.get("camera_prompt_on_stable", False))
    interaction_enabled = bool(interactive and sys.stdin.isatty())
    read_failures = 0
    last_canvas = None
    last_focus = None
    last_status_lines = []

    try:
        while True:
            action = ui.poll_action(delay_ms=1)
            if action == "quit":
                break
            if action == "toggle_pause":
                paused = not paused
            elif action == "prompt":
                pending_prompt = True
            elif action == "toggle_help":
                ui.toggle_help()

            if paused:
                if last_canvas is not None:
                    ui.render(last_canvas, last_status_lines, last_focus)
                time.sleep(0.05)
                continue

            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
                read_failures += 1
                if read_failures >= max(1, int(cfg.camera.get("read_retry_limit", 30))):
                    break
                time.sleep(max(0.0, float(cfg.camera.get("read_retry_delay_ms", 50)) / 1000.0))
                continue
            read_failures = 0
            frame_index += 1
            if frame_index % stride != 0:
                if last_canvas is not None:
                    ui.render(last_canvas, last_status_lines, last_focus)
                continue

            decision = pipeline.process_frame(frame_bgr, frame_index=frame_index)
            step_text = f"STEP: {decision.display_step} ({decision.display_conf:.2f})"
            canvas = draw_overlay(frame_bgr, decision.perception.fused_detections, step_text)
            focus_crop = safe_crop(frame_bgr, decision.perception.relevant_detections, pad=10)
            status_lines = [
                f"camera={int(camera_index)} backend={backend_name} frame={frame_index}",
                f"stable={int(decision.is_stable)} stable_count={decision.stable_count}",
                f"fused={decision.step_prediction.fused_step} s={decision.step_prediction.state_step} r={decision.step_prediction.retrieval_step}",
            ]

            if writer is not None:
                writer.write(canvas)
            artifacts.save_snapshot(canvas, decision.iter_index, frame_index)
            artifacts.log_iteration(decision)

            last_canvas = canvas
            last_focus = focus_crop
            last_status_lines = status_lines
            ui.render(canvas, status_lines, focus_crop)

            if interaction_enabled and decision.is_stable and pending_prompt:
                paused = True
                handle_stable_console_interaction(pipeline, decision, artifacts, topology=topology)
                pending_prompt = False
                paused = False
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        ui.close()
        artifacts.finalize()

    return artifacts.run_dir
