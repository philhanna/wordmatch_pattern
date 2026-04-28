"""Application-layer training use case.

This module is the boundary between the outer layers (CLI adapters, tests)
and the domain.  It accepts abstract ports so that callers never depend
directly on file I/O or other infrastructure concerns.
"""
from __future__ import annotations

from pattern.domain import Model, build_model
from pattern.ports import WordSource


def train(source: WordSource, smoothing_k: float = 0.5) -> Model:
    """Build a :class:`~pattern.domain.model.Model` from the given word source.

    Parameters
    ----------
    source:
        Any :class:`~pattern.ports.WordSource` implementation.  The words are
        consumed exactly once.
    smoothing_k:
        Laplace smoothing constant forwarded to
        :func:`~pattern.domain.training.build_model`.  Defaults to 0.5.

    Returns
    -------
    Model
        A trained model ready for probability queries.
    """
    return build_model(source.words(), smoothing_k)
