# pattern.ports
"""Abstract port interfaces for the pattern package.

Ports define *what* the domain needs from the outside world without
specifying *how* those needs are satisfied.  Concrete adapters in
:mod:`pattern.adapters` implement these interfaces.

Exported names
--------------
:class:`WordSource`
    Supplies a stream of raw words for training.
:class:`ModelStore`
    Saves and loads trained :class:`~pattern.domain.model.Model` instances.
"""
from pattern.ports.word_source import WordSource
from pattern.ports.model_store import ModelStore

__all__ = ["WordSource", "ModelStore"]
