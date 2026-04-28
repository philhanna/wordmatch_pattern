# pattern.domain
"""Core domain logic: model data structure, training, and inference.

This package is intentionally free of I/O dependencies.  Everything here
operates on plain Python objects, making it straightforward to test in
isolation.

Exported names
--------------
:class:`Model`
    The trained model data structure.
:func:`build_model`
    Constructs a :class:`Model` from an iterable of raw words.
:func:`match_probability`
    Estimates P(at least one word matches a pattern).
:func:`expected_match_count`
    Returns the expected number of matching words for a pattern.
"""
from pattern.domain.model import Model
from pattern.domain.training import build_model
from pattern.domain.inference import match_probability, expected_match_count

__all__ = ["Model", "build_model", "match_probability", "expected_match_count"]
