from __future__ import annotations

from pathlib import Path

from pattern.adapters.json_model_store import JsonModelStore
from pattern.domain import match_probability


def query(model_path: str | Path, pattern: str) -> float:
    model = JsonModelStore(model_path).load()
    return match_probability(model, pattern)
