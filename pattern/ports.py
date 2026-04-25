from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .domain import Model


class WordSource(ABC):
    """Port: provides a stream of raw word strings to train on."""

    @abstractmethod
    def words(self) -> Iterable[str]: ...


class ModelStore(ABC):
    """Port: persists and retrieves a trained Model."""

    @abstractmethod
    def save(self, model: Model) -> None: ...

    @abstractmethod
    def load(self) -> Model: ...
