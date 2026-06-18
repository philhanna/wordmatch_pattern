"""Abstract port for model persistence.

A *model store* is anything that can save and reload a trained
:class:`~pattern.domain.model.Model`.  Decoupling persistence behind this
interface means the rest of the application has no dependency on a particular
serialisation format (JSON, pickle, a database, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pattern.domain import Model


class ModelStore(ABC):
    """Port: persists and retrieves a trained :class:`~pattern.domain.model.Model`.

    Concrete adapters (e.g. :class:`~pattern.adapters.joblib_model_store.JoblibModelStore`)
    implement the two abstract methods to bind the port to a specific storage
    backend.
    """

    @abstractmethod
    def save(self, model: Model) -> None:
        """Persist *model* to the backing store.

        Parameters
        ----------
        model:
            The trained model to save.  Any previously stored model at the
            same location should be overwritten.
        """
        ...

    @abstractmethod
    def load(self) -> Model:
        """Load and return the model from the backing store.

        Returns
        -------
        Model
            The previously saved model.

        Raises
        ------
        FileNotFoundError
            If no model has been saved at the configured location.
        """
        ...
