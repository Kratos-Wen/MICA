"""Configuration helpers for the modular MICA package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """Lightweight wrapper around the YAML payload."""

    raw: Dict[str, Any]

    @property
    def video(self) -> Dict[str, Any]:
        return self.raw.get("video", {})

    @property
    def camera(self) -> Dict[str, Any]:
        return self.raw.get("camera", {})

    @property
    def detection(self) -> Dict[str, Any]:
        return self.raw.get("detection", {})

    @property
    def fusion(self) -> Dict[str, Any]:
        return self.raw.get("fusion", {})

    @property
    def depth_context(self) -> Dict[str, Any]:
        return self.raw.get("depth_context", {})

    @property
    def stability(self) -> Dict[str, Any]:
        return self.raw.get("stability", {})

    @property
    def asf(self) -> Dict[str, Any]:
        return self.raw.get("asf", {})

    @property
    def gallery(self) -> Dict[str, Any]:
        return self.raw.get("gallery", {})

    @property
    def llm(self) -> Dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def agent(self) -> Dict[str, Any]:
        return self.raw.get("agent", {})

    @property
    def interaction(self) -> Dict[str, Any]:
        return self.raw.get("interaction", {})

    @property
    def ablation(self) -> Dict[str, Any]:
        return self.raw.get("ablation", {})

    @property
    def runlog(self) -> Dict[str, Any]:
        return self.raw.get("runlog", {})

    @property
    def safety(self) -> Dict[str, Any]:
        return self.raw.get("safety", {})


def load_config(path: Optional[str]) -> Config:
    """Load a YAML file, or the package example config when omitted."""

    if path is None:
        path = str(Path(__file__).resolve().parent / "resources" / "config.example.yaml")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Config(raw=payload)
