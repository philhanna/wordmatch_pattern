# pattern.adapters
"""Concrete implementations of the port interfaces.

Each adapter wires the abstract port contract to a specific I/O mechanism:

:class:`FileWordSource`
    Reads words from a plain-text file, one per line.
:class:`JsonModelStore`
    Persists and loads a trained :class:`~pattern.domain.model.Model` as a
    JSON file.
"""
from pattern.adapters.file_word_source import FileWordSource
from pattern.adapters.json_model_store import JsonModelStore

__all__ = ["FileWordSource", "JsonModelStore"]
