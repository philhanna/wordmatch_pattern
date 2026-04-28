"""JSON-file implementation of the :class:`~pattern.ports.ModelStore` port.

Serialises and deserialises a :class:`~pattern.domain.model.Model` as a
human-readable JSON file.  This makes trained models easy to inspect,
version-control, and transfer without any binary tooling.
"""
from __future__ import annotations

import json
from pathlib import Path

from pattern.domain import Model
from pattern.ports import ModelStore


class JsonModelStore(ModelStore):
    """Adapter: saves and loads a :class:`~pattern.domain.model.Model` as a JSON file.

    Parameters
    ----------
    path:
        Filesystem path for the JSON model file.  Accepts both :class:`str`
        and :class:`~pathlib.Path`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, model: Model) -> None:
        """Serialise *model* and write it to :attr:`path`.

        The file is written atomically within the constraints of
        :func:`open`; any existing file at :attr:`path` is overwritten.
        Output is written in a compact JSON form to minimize file size while
        remaining human-readable.

        Parameters
        ----------
        model:
            The trained model to persist.
        """
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(model.to_dict(), fh, separators=(",", ":"))

    def load(self) -> Model:
        """Read the JSON file at :attr:`path` and return a reconstructed model.

        Returns
        -------
        Model
            The deserialised model.

        Raises
        ------
        FileNotFoundError
            If :attr:`path` does not exist.
        """
        with open(self.path, encoding="utf-8") as fh:
            return Model.from_dict(json.load(fh))
