"""Ultralytics YOLO wrapper."""
from __future__ import annotations
from typing import List, Dict, Any, Union
import numpy as np

from typing import List, Dict, Any, Union
import numpy as np
import cv2
import torch
from torchvision.ops import nms

class YOLODetector:
    """
    Ultralytics YOLO wrapper with explicit per-frame NMS and optional TTA.
    """
    def __init__(self,
                 weights: str,
                 device: Union[int, str] = "cpu",
                 nms_iou: float = 0.5,
                 agnostic_nms: bool = True,
                 max_det: int = 300,
                 use_builtin_tta: bool = False,
                 tta_scales: List[float] = [0.83, 1.20],
                 tta_hflip: bool = True):
        """
        Args:
            weights: Path to YOLO .pt weights.
            device: CUDA index or 'cpu'.
            nms_iou: IoU threshold for NMS.
            agnostic_nms: Class-agnostic NMS inside Ultralytics.
            max_det: Max detections per image (Ultralytics).
            use_builtin_tta: If True, call Ultralytics' internal TTA (augment=True).
            tta_scales: Multi-scale factors for manual TTA.
            tta_hflip: Whether to add horizontal flip for manual TTA.
        """
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device = device
        self.nms_iou = float(nms_iou)
        self.agnostic_nms = bool(agnostic_nms)
        self.max_det = int(max_det)
        self.use_builtin_tta = bool(use_builtin_tta)
        self.tta_scales = list(tta_scales or [])
        self.tta_hflip = bool(tta_hflip)

    # ---------- internal helpers ----------
    def _ultra_predict(self, frame_bgr: np.ndarray, conf: float, iou: float,
                       augment: bool = False) -> List[Dict[str, Any]]:
        """
        One-shot YOLO predict -> list of det dicts (already in original image coordinates).
        """
        results = self.model.predict(
            frame_bgr,
            conf=conf,
            iou=iou,                 # Ultralytics internal NMS IoU
            agnostic_nms=self.agnostic_nms,
            max_det=self.max_det,
            augment=augment,         # built-in TTA if True
            verbose=False,
            device=self.device
        )
        dets: List[Dict[str, Any]] = []
        for r in results:
            names = getattr(r, "names", None)
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy
            cls  = r.boxes.cls
            confs= r.boxes.conf
            if xyxy is None or cls is None:
                continue
            xyxy = xyxy.detach().cpu().numpy()
            cls  = cls.detach().cpu().numpy().astype(int)
            confs= confs.detach().cpu().numpy() if confs is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
            for i, box in enumerate(xyxy):
                c = int(cls[i]); cf = float(confs[i])
                if isinstance(names, dict):
                    cname = names.get(c, str(c))
                elif isinstance(names, (list, tuple)) and 0 <= c < len(names):
                    cname = names[c]
                else:
                    cname = str(c)
                dets.append({"xyxy": box.tolist(), "name": str(cname), "conf": cf})
        return dets

    @staticmethod
    def _torch_nms(dets: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
        """
        Class-agnostic NMS as a second-pass safety net.
        """
        if not dets:
            return dets
        boxes = torch.tensor([d["xyxy"] for d in dets], dtype=torch.float32)
        scores= torch.tensor([d["conf"] for d in dets], dtype=torch.float32)
        keep  = nms(boxes, scores, float(iou_thr)).tolist()
        return [dets[i] for i in keep]

    @staticmethod
    def _clip_box(b, w, h):
        x1,y1,x2,y2 = b
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return [x1,y1,x2,y2]

    # ---------- public API ----------
    def detect(self, frame_bgr: np.ndarray, conf: float = 0.25, tta: bool = False) -> List[Dict[str, Any]]:
        """
        Run detection on a single BGR frame with NMS and optional TTA.

        Args:
            frame_bgr: BGR image.
            conf: Confidence threshold.
            tta: If True and use_builtin_tta=False, run manual TTA (flip + multiscale).

        Returns:
            List of det dicts: {'xyxy':[x1,y1,x2,y2], 'name':str, 'conf':float}
        """
        H, W = frame_bgr.shape[:2]

        # 方式 1：直接用 Ultralytics 内置 TTA（简单）
        if self.use_builtin_tta and tta:
            dets = self._ultra_predict(frame_bgr, conf=conf, iou=self.nms_iou, augment=True)
            # 二次保险 NMS
            dets = self._torch_nms(dets, self.nms_iou)
            return dets

        # 方式 2：单图 + 手工 TTA（可控）
        all_dets: List[Dict[str, Any]] = []

        # 原图
        base = self._ultra_predict(frame_bgr, conf=conf, iou=self.nms_iou, augment=False)
        all_dets.extend(base)

        if tta:
            # 水平翻转
            if self.tta_hflip:
                img = cv2.flip(frame_bgr, 1)
                dets_flip = self._ultra_predict(img, conf=conf, iou=self.nms_iou, augment=False)
                # 映射回原图坐标
                for d in dets_flip:
                    x1,y1,x2,y2 = d["xyxy"]
                    d["xyxy"] = self._clip_box([W - x2, y1, W - x1, y2], W, H)
                all_dets.extend(dets_flip)

            # 多尺度
            for s in self.tta_scales:
                if abs(s-1.0) < 1e-3:
                    continue
                new_w, new_h = int(W*s), int(H*s)
                img = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                dets_scale = self._ultra_predict(img, conf=conf, iou=self.nms_iou, augment=False)
                # 映射回原图坐标
                for d in dets_scale:
                    x1,y1,x2,y2 = d["xyxy"]
                    d["xyxy"] = self._clip_box([x1/s, y1/s, x2/s, y2/s], W, H)
                all_dets.extend(dets_scale)

        # 合并后统一 NMS（class-agnostic）
        dets = self._torch_nms(all_dets, self.nms_iou)
        return dets
