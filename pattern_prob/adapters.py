from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .domain import Model
from .ports import ModelStore, WordSource


class FileWordSource(WordSource):
    """Adapter: reads words from a flat text file (one word per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def words(self) -> Iterable[str]:
        with open(self.path, encoding="utf-8", errors="ignore") as fh:
            yield from fh


class JsonModelStore(ModelStore):
    """Adapter: saves and loads a Model as a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, model: Model) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(model.to_dict(), fh)

    def load(self) -> Model:
        with open(self.path, encoding="utf-8") as fh:
            return Model.from_dict(json.load(fh))
