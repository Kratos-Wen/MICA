"""Source opening helpers for offline and live inputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2


def discover_yolo_weights(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    candidates = [
        Path(__file__).resolve().parents[2] / "best.pt",
        Path.cwd() / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("YOLO weights not provided and no best.pt was found.")


def _camera_backend_candidates(name: str) -> list[Tuple[str, Optional[int]]]:
    normalized = str(name or "auto").strip().lower()
    lookup = {
        "default": ("default", None),
        "auto": ("default", None),
        "dshow": ("dshow", getattr(cv2, "CAP_DSHOW", None)),
        "msmf": ("msmf", getattr(cv2, "CAP_MSMF", None)),
        "v4l2": ("v4l2", getattr(cv2, "CAP_V4L2", None)),
        "ffmpeg": ("ffmpeg", getattr(cv2, "CAP_FFMPEG", None)),
    }
    if normalized in {"", "auto"}:
        ordered = [
            lookup["default"],
            lookup["dshow"],
            lookup["msmf"],
            lookup["v4l2"],
            lookup["ffmpeg"],
        ]
    else:
        ordered = [lookup.get(normalized, (normalized, None)), lookup["default"]]
    deduped: list[Tuple[str, Optional[int]]] = []
    seen: set[Tuple[str, Optional[int]]] = set()
    for item in ordered:
        if item[1] is None and item[0] not in {"default", "auto"} and normalized != item[0]:
            continue
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def open_camera_capture(index: int, camera_cfg: dict) -> Tuple[cv2.VideoCapture, str]:
    last_error = f"Failed to open camera: {index}"
    for backend_name, backend_id in _camera_backend_candidates(str(camera_cfg.get("backend", "auto"))):
        capture = cv2.VideoCapture(index) if backend_id is None else cv2.VideoCapture(index, backend_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera_cfg.get("width", 1280)))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera_cfg.get("height", 720)))
        capture.set(cv2.CAP_PROP_FPS, float(camera_cfg.get("fps", 30.0)))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, float(camera_cfg.get("buffer_size", 1)))
        if capture.isOpened():
            warmup_frames = max(0, int(camera_cfg.get("warmup_frames", 0)))
            warmup_delay_sec = max(0.0, float(camera_cfg.get("warmup_delay_ms", 0)) / 1000.0)
            ready = 0
            for _ in range(warmup_frames):
                ok, frame_bgr = capture.read()
                if ok and frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
                    ready += 1
                if warmup_delay_sec > 0:
                    time.sleep(warmup_delay_sec)
            if warmup_frames == 0 or ready > 0:
                return capture, backend_name
        capture.release()
        last_error = f"Failed to open camera: {index} with backend={backend_name}"
    raise RuntimeError(last_error)
