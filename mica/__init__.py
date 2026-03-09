"""Paper-aligned modular open-source package for MICA."""

from mica.config import Config, load_config
from mica.runtime.live_runner import run_camera
from mica.runtime.offline_runner import run_video

__all__ = ["Config", "load_config", "run_camera", "run_video"]
