from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class WordSource(ABC):
    """Port: provides a stream of raw word strings to train on."""

    @abstractmethod
    def words(self) -> Iterable[str]: ...
