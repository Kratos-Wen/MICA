"""OpenCV live UI for the camera runner."""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


class LiveUI:
    """Render frames and map keyboard input to simple runtime actions."""

    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "MICA Live",
        focus_window_name: str = "MICA Focus",
        show_focus: bool = True,
        show_help: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.window_name = str(window_name)
        self.focus_window_name = str(focus_window_name)
        self.show_focus = bool(show_focus)
        self.show_help = bool(show_help)
        self._opened = False

    def open(self) -> None:
        if not self.enabled or self._opened:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.show_focus:
            cv2.namedWindow(self.focus_window_name, cv2.WINDOW_NORMAL)
        self._opened = True

    def render(
        self,
        canvas: np.ndarray,
        status_lines: Optional[Iterable[str]] = None,
        focus_crop: Optional[np.ndarray] = None,
    ) -> None:
        if not self.enabled:
            return
        self.open()
        frame = canvas.copy()
        y = 58
        for line in [str(item) for item in (status_lines or []) if str(item).strip()][:8]:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 20
        if self.show_help:
            cv2.putText(
                frame,
                "Q quit | P pause | F prompt | H help",
                (10, max(24, frame.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (120, 255, 180),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.window_name, frame)
        if self.show_focus and focus_crop is not None and focus_crop.size > 0:
            cv2.imshow(self.focus_window_name, focus_crop)

    def poll_action(self, delay_ms: int = 1) -> str:
        if not self.enabled:
            return ""
        self.open()
        key = cv2.waitKey(int(delay_ms)) & 0xFF
        if key in {255, -1}:
            return ""
        if key == ord("q"):
            return "quit"
        if key in {ord("p"), ord(" ")}:
            return "toggle_pause"
        if key == ord("f"):
            return "prompt"
        if key == ord("h"):
            return "toggle_help"
        return ""

    def toggle_help(self) -> bool:
        self.show_help = not self.show_help
        return self.show_help

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        if self.show_focus:
            try:
                cv2.destroyWindow(self.focus_window_name)
            except Exception:
                pass
        self._opened = False
