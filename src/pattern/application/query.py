from __future__ import annotations

from pathlib import Path

from pattern.adapters.json_model_store import JsonModelStore
from pattern.domain.model import Model


def load_model(model_path: str | Path) -> Model:
    return JsonModelStore(model_path).load()
