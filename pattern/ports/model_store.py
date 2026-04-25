from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pattern.domain import Model


class ModelStore(ABC):
    """Port: persists and retrieves a trained Model."""

    @abstractmethod
    def save(self, model: Model) -> None: ...

    @abstractmethod
    def load(self) -> Model: ...
