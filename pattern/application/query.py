"""Application-layer model-loading use case.

Provides a thin convenience function that hides the concrete adapter choice
(joblib file store) from callers that only need to load an already-trained
model by path.
"""
from __future__ import annotations

from pathlib import Path

from pattern.adapters.joblib_model_store import JoblibModelStore
from pattern.domain.model import Model


def load_model(model_path: str | Path) -> Model:
    """Load a trained model from a joblib file on disk.

    Parameters
    ----------
    model_path:
        Path to the joblib model file produced by the ``train`` command.
        Accepts both :class:`str` and :class:`~pathlib.Path`.

    Returns
    -------
    Model
        The deserialised model, ready for
        :func:`~pattern.domain.inference.match_probability` calls.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    """
    return JoblibModelStore(model_path).load()
