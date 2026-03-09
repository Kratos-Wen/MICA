"""Depth model stub (replace with Depth-Anything for production)."""
from __future__ import annotations
from typing import Optional
import numpy as np

class DepthStub:
    """Minimal depth estimator returning a horizontal gradient."""
    def __init__(self, device: Optional[str] = "cpu"):
        """Initialize the stub."""
        self.device = device

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a synthetic depth map in [0,1].

        Args:
            frame_bgr: BGR image.

        Returns:
            Depth map (H,W), smaller = nearer.
        """
        h, w = frame_bgr.shape[:2]
        depth = np.tile(np.linspace(0.2, 1.0, w, dtype=np.float32), (h, 1))
        return depth
