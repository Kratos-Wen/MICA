"""Configuration utilities for the MICA-Glasses pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

@dataclass
class Config:
    """Container for parsed configuration values."""
    raw: Dict[str, Any]

    @property
    def video(self) -> Dict[str, Any]: return self.raw.get("video", {})
    @property
    def detection(self) -> Dict[str, Any]: return self.raw.get("detection", {})
    @property
    def fusion(self) -> Dict[str, Any]: return self.raw.get("fusion", {})
    @property
    def depth_context(self) -> Dict[str, Any]: return self.raw.get("depth_context", {})
    @property
    def stability(self) -> Dict[str, Any]: return self.raw.get("stability", {})
    @property
    def asf(self) -> Dict[str, Any]: return self.raw.get("asf", {})
    @property
    def gallery(self) -> Dict[str, Any]: return self.raw.get("gallery", {})
    @property
    def runlog(self) -> Dict[str, Any]: return self.raw.get("runlog", {})

def load_config(path: Optional[str]) -> Config:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file. If None, loads the package example config.

    Returns:
        Config: Parsed configuration wrapper.
    """
    if path is None:
        path = str(Path(__file__).resolve().parents[1] / "resources" / "config.example.yaml")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config(raw=data)
