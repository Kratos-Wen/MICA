"""Depth-guided object context selection (Eq. 1)."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np, math

def center_of(box: List[float]) -> Tuple[float,float]:
    """Return (cx,cy) for [x1,y1,x2,y2]."""
    x1,y1,x2,y2 = box
    return 0.5*(x1+x2), 0.5*(y1+y2)

def depth_at(depth: np.ndarray, cx: float, cy: float) -> float:
    """Sample depth at floating coords using nearest neighbor."""
    h,w = depth.shape[:2]
    x = int(np.clip(cx,0,w-1)); y = int(np.clip(cy,0,h-1))
    return float(depth[y,x])

def select_by_depth(
    dets: List[Dict[str,Any]], depth: np.ndarray, tau_p: float=80.0, tau_d: float=0.12
) -> Tuple[List[Dict[str,Any]], Optional[int]]:
    """
    Select local context objects around the nearest object.

    Args:
        dets: Detection dicts with 'xyxy' and 'name'.
        depth: Depth map in [0,1], smaller means nearer.
        tau_p: Pixel distance threshold.
        tau_d: Depth difference threshold.

    Returns:
        (selected_dets, nearest_index)
    """
    if not dets: return [], None
    info=[]
    for i,d in enumerate(dets):
        cx,cy = center_of(d["xyxy"])
        di = depth_at(depth,cx,cy)
        info.append((i,cx,cy,di))
    nearest_idx, nx, ny, nd = min(info, key=lambda t:t[3])
    selected=[]
    for i,cx,cy,di in info:
        if math.hypot(cx-nx, cy-ny) <= float(tau_p) and abs(di-nd) <= float(tau_d):
            selected.append(dets[i])
    return selected, nearest_idx
