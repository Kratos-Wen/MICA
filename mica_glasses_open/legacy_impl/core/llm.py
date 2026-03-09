"""Local LLM client (Ollama) and a null fallback."""
from __future__ import annotations
from typing import Optional, Dict, Any, List
import requests

class LLM:
    """
    Simple Ollama client. If the server is not reachable, it falls back to a null mode.
    """
    def __init__(self, model: str = "qwen2.5:7b-instruct", url: str = "http://localhost:11434/api/generate",
                 temperature: float = 0.2, max_tokens: int = 256, timeout: int = 30):
        self.model = model
        self.url = url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self._disabled = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate text with LLM. Returns a short deterministic string if server is unavailable.
        """
        if self._disabled:
            return self._null(system_prompt, user_prompt)
        payload = {
            "model": self.model,
            "prompt": user_prompt or "",
            "system": system_prompt or "", #"prompt": f"<<SYS>>{system_prompt}\n<<USR>>{user_prompt}",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 32768,
                "top_p": 0.9,
            }
        }
        try:
            r = requests.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            self._disabled = True
            return self._null(system_prompt, user_prompt)

    @staticmethod
    def _null(system_prompt: str, user_prompt: str) -> str:
        """Fallback text when LLM is not reachable."""
        return "[LLM offline] " + (user_prompt[:200] if user_prompt else "")
