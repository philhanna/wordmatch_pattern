from __future__ import annotations

from ..domain.model import Model
from ..domain.training import build_model
from ..ports.word_source import WordSource


def train(source: WordSource, smoothing_k: float = 0.5) -> Model:
    """Build a Model from the given WordSource."""
    return build_model(source.words(), smoothing_k)
