# pattern.application
"""Application-layer use cases.

These functions orchestrate the domain and adapters to implement concrete
workflows.  They are the preferred entry points for code outside this package;
they insulate callers from internal restructuring.

Exported names
--------------
:func:`train`
    Build a model from a :class:`~pattern.ports.WordSource`.
:func:`load_model`
    Load a previously trained model from disk.
:func:`match_probability`
    Re-exported from the domain for convenience.
"""
from pattern.application.train import train
from pattern.application.query import load_model
from pattern.domain.inference import match_probability

__all__ = ["train", "load_model", "match_probability"]
