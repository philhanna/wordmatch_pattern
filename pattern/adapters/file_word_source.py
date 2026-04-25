from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pattern.ports import WordSource


class FileWordSource(WordSource):
    """Adapter: reads words from a flat text file (one word per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def words(self) -> Iterable[str]:
        with open(self.path, encoding="utf-8", errors="ignore") as fh:
            yield from fh
