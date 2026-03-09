"""Rolling conversation memory buffer for agents."""
from __future__ import annotations
from typing import List, Dict, Any

class Memory:
    """
    Simple rolling memory for user <-> assistant turns.
    Keeps the last N messages as plain text.
    """
    def __init__(self, max_items: int = 20):
        self.max_items = int(max_items)
        self.meta: Dict[str, Any] = {}
        self._items: List[Dict[str, str]] = []

    def append(self, role: str, text: str) -> None:
        self._items.append({"role": role, "text": text})
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items:]

    def render_text(self) -> str:
        """
        Render memory as a readable text block for LLM prompts.
        """
        return "\n".join([f"{it['role'].upper()}: {it['text']}" for it in self._items])
