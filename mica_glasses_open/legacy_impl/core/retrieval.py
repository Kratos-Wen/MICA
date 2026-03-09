# mica_glasses/core/retrieval.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re
import cv2
import numpy as np

# ---- optional CLIP backend ----
_HAS_CLIP = False
try:
    import torch
    import open_clip
    from PIL import Image
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # both 1-D; return cosine sim
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def _embed_rgb_mean(img_bgr: np.ndarray, size: int = 8) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    vec = img.astype(np.float32).reshape(-1) / 255.0
    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
    return vec.astype(np.float32)

class ClipBackbone:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
                 device: Optional[str] = None):
        if not _HAS_CLIP:
            raise RuntimeError("CLIP mode requires 'open_clip_torch' and 'Pillow'.")
        dev = device or ("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.device = torch.device(dev)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        # 输出维度（通常 512）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            feat = self.model.encode_image(dummy)
            self.dim = int(feat.shape[-1])

    @torch.no_grad()
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()  # (dim,)

class GalleryIndex:
    """
    Reference gallery organized as folders:
        root/Step1_xxxx/*.jpg, root/Step2_xxxx/*.png, ...
    Parse step id from folder name: r"^Step(?P<step>\\d+)_?(?P<profile>[A-Za-z0-9]*)$".
    """
    GALLERY_RX = re.compile(r"^Step(?P<step>\d+)_?(?P<profile>[A-Za-z0-9]*)$")

    def __init__(self, root: str, exts: List[str], embed_mode: str = "rgb-mean-8"):
        self.root = Path(root)
        self.exts = {str(e).lower() for e in (exts or [])}
        self.mode = (embed_mode or "rgb-mean-8").strip().lower()
        # 兼容旧写法 "rgb-mean-8x8"
        m_old = re.match(r"rgb-mean-(\d+)x\1$", self.mode)
        if m_old:
            self.mode = f"rgb-mean-{m_old.group(1)}"

        self.clip: Optional[ClipBackbone] = None
        self.rgb_size: Optional[int] = None
        if self.mode == "clip":
            if not _HAS_CLIP:
                raise RuntimeError("embed=clip requires 'open_clip_torch' and 'Pillow' installed.")
            self.clip = ClipBackbone()
            self.dim = self.clip.dim
            self.backend = f"clip/{self.clip.device}"
        else:
            m = re.match(r"rgb-mean-(\d+)$", self.mode)
            self.rgb_size = int(m.group(1)) if m else 8
            self.dim = self.rgb_size * self.rgb_size * 3
            self.backend = f"rgb-mean-{self.rgb_size}"

        self.items: List[Dict[str, Any]] = []  # {step, profile, path, emb}

    def _parse_folder(self, folder: Path) -> Optional[Tuple[str, str]]:
        m = self.GALLERY_RX.match(folder.name)
        if not m: return None
        step = f"S{int(m.group('step'))}"
        profile = m.group('profile') or ""
        return step, profile

    def _embed(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.clip:
            return self.clip.embed(img_bgr)
        return _embed_rgb_mean(img_bgr, size=int(self.rgb_size or 8))

    def build(self) -> int:
        if not self.root.exists():
            raise FileNotFoundError(f"Gallery root not found: {self.root}")
        count = 0
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir(): continue
            parsed = self._parse_folder(sub)
            if not parsed: continue
            step, profile = parsed
            for p in sub.rglob("*"):
                if p.is_file() and p.suffix.lower() in self.exts:
                    img = cv2.imread(str(p))
                    if img is None: continue
                    emb = self._embed(img)
                    # 归一化，便于后续直接点乘当余弦
                    emb = emb.astype(np.float32)
                    n = np.linalg.norm(emb) + 1e-12
                    emb = emb / n
                    self.items.append({"step": step, "profile": profile, "path": str(p), "emb": emb})
                    count += 1
        # 关键：打印出来确认分支
        dev = self.clip.device if self.clip else "cpu"
        print(f"[Gallery] mode={self.mode} dim={self.dim} device={dev} N={count}")
        return count

    def predict(self, frame_bgr: np.ndarray, topk: int = 5) -> Tuple[str, float, Dict[str, Any]]:
        if not self.items:
            return "S1", 0.5, {"reason":"empty_gallery"}
        q = self._embed(frame_bgr).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        mats = np.stack([it["emb"] for it in self.items], axis=0)  # (N, D)
        sims = (mats @ q)  # 余弦相似（emb 已单位化）
        # topk
        idxs = sims.argsort()[::-1][:max(1, topk)]
        hist = {}
        for i in idxs:
            st = self.items[i]["step"]
            hist[st] = hist.get(st, 0) + 1
        pred_step = max(hist.items(), key=lambda x: x[1])[0]
        conf = float(np.mean(sims[idxs]))  # 平均相似度作为置信度
        profiles = [self.items[i]["profile"] for i in idxs]
        return pred_step, conf, {"topk": int(topk), "profiles": profiles, "hist": hist}
