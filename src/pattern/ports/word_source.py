"""Abstract port for word sources.

A *word source* is anything that can supply a stream of raw word strings to
:func:`~pattern.domain.training.build_model`.  Defining this as an abstract
base class lets application code depend on the interface rather than a concrete
I/O mechanism, making it straightforward to swap file-based sources for
database queries, network streams, or in-memory lists.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class WordSource(ABC):
    """Port: provides a stream of raw word strings to train on.

    Implementors yield one word per element.  Words may include surrounding
    whitespace, mixed case, or non-alphabetic characters; the training layer
    normalises and filters them.
    """

    @abstractmethod
    def words(self) -> Iterable[str]:
        """Return an iterable of raw word strings.

        The iterable need only be consumed once.  Each element is a single
        word (or line), which the caller will strip and validate.

        Returns
        -------
        Iterable[str]
            Raw word strings in any order.
        """
        ...
