"""Shared KB and filesystem helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    ensure_dir(path.parent)


def kb_components_list(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accept both list-based and dict-based KB layouts."""

    components = kb.get("components")
    if isinstance(components, list):
        return [item for item in components if isinstance(item, dict) and "name" in item]

    out: List[Dict[str, Any]] = []
    skip_keys = {"aliases", "workflow", "config", "meta"}
    for key, value in kb.items():
        if key in skip_keys or not isinstance(value, dict):
            continue
        record = dict(value)
        record["name"] = str(key)
        out.append(record)
    return out


def kb_index_by_name(components: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lowercase name index for KB components."""

    index: Dict[str, Dict[str, Any]] = {}
    for component in components:
        name = str(component.get("name", "")).strip().lower()
        if name:
            index[name] = component
    return index
