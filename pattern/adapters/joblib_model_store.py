"""joblib-file implementation of the :class:`~pattern.ports.ModelStore` port.

Serialises and deserialises a :class:`~pattern.domain.model.Model` with
:mod:`joblib`.  joblib pickles the model object directly, which is both faster
and more compact for the large nested frequency tables than a text JSON
representation, and it requires no manual key conversion on the model.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from pattern.domain import Model
from pattern.ports import ModelStore


class JoblibModelStore(ModelStore):
    """Adapter: saves and loads a :class:`~pattern.domain.model.Model` with joblib.

    Parameters
    ----------
    path:
        Filesystem path for the joblib model file.  Accepts both :class:`str`
        and :class:`~pathlib.Path`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, model: Model) -> None:
        """Serialise *model* and write it to :attr:`path`.

        Any existing file at :attr:`path` is overwritten.

        Parameters
        ----------
        model:
            The trained model to persist.
        """
        joblib.dump(model, self.path)

    def load(self) -> Model:
        """Read the joblib file at :attr:`path` and return the model.

        Returns
        -------
        Model
            The deserialised model.

        Raises
        ------
        FileNotFoundError
            If :attr:`path` does not exist.
        """
        return joblib.load(self.path)
