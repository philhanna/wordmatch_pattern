# pattern.adapters
"""Concrete implementations of the port interfaces.

Each adapter wires the abstract port contract to a specific I/O mechanism:

:class:`FileWordSource`
    Reads words from a plain-text file, one per line.
:class:`JoblibModelStore`
    Persists and loads a trained :class:`~pattern.domain.model.Model` with
    joblib.
"""
from pattern.adapters.file_word_source import FileWordSource
from pattern.adapters.joblib_model_store import JoblibModelStore

__all__ = ["FileWordSource", "JoblibModelStore"]
