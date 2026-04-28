# pattern
"""Top-level package for the *pattern* library.

The package exposes the complete public API through a flat namespace so that
typical callers only need a single import:

.. code-block:: python

    from pattern import train, match_probability, FileWordSource

Architecture
------------
The code is organised in three layers following a ports-and-adapters
(hexagonal) style:

* **domain** — pure business logic: the :class:`~pattern.domain.model.Model`
  data structure, frequency-table training, and probability inference.
* **ports** — abstract interfaces (:class:`~pattern.ports.WordSource`,
  :class:`~pattern.ports.ModelStore`) that the domain depends on.
* **adapters** — concrete I/O implementations
  (:class:`~pattern.adapters.FileWordSource`,
  :class:`~pattern.adapters.JsonModelStore`) that satisfy the port interfaces.
* **application** — thin use-case functions (:func:`~pattern.application.train`,
  :func:`~pattern.application.load_model`) that wire domain and adapters
  together.

Public API
----------
"""
from pattern.domain import Model, match_probability, expected_match_count
from pattern.application import train
from pattern.adapters import FileWordSource, JsonModelStore
from pattern.ports import WordSource, ModelStore

__all__ = [
    "Model",
    "train",
    "match_probability",
    "expected_match_count",
    "FileWordSource",
    "JsonModelStore",
    "WordSource",
    "ModelStore",
]
