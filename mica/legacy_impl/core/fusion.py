"""Multi-frame detection fusion (confidence-weighted IoU)."""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

def iou(a, b) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    areaA = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    areaB = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = areaA + areaB - inter + 1e-6
    return inter/union

def fuse_window(frames: List[List[Dict[str, Any]]], iou_thr: float=0.5) -> List[Dict[str, Any]]:
    """
    Fuse detections across several processed iterations.

    Args:
        frames: List of detection lists over time.
        iou_thr: IoU threshold to merge boxes of the same class.

    Returns:
        Fused detections.
    """
    acc = []
    for dets in frames:
        for d in dets:
            acc.append((str(d["name"]).lower(), d["xyxy"], float(d.get("conf",0.0))))
    by_name = {}
    for name, box, conf in acc:
        by_name.setdefault(name, []).append((box, conf))

    fused = []
    for name, items in by_name.items():
        used = [False]*len(items)
        for i,(bi,ci) in enumerate(items):
            if used[i]: continue
            group=[(bi,ci)]; used[i]=True
            for j in range(i+1,len(items)):
                if used[j]: continue
                bj,cj = items[j]
                if iou(bi,bj) >= iou_thr:
                    group.append((bj,cj)); used[j]=True
            ws = np.array([c for _,c in group], dtype=np.float32)
            boxes = np.array([b for b,_ in group], dtype=np.float32)
            wsum = float(ws.sum())+1e-6
            b_f = (boxes * ws[:,None]).sum(axis=0)/wsum
            c_f = float(ws.mean())
            fused.append({"name": name, "xyxy": b_f.tolist(), "conf": c_f})
    return fused
