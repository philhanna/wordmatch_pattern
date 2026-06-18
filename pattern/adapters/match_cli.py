"""Command-line interface for querying a trained pattern-probability model.

Loads a model produced by the ``train`` command and prints the match
probability, expected match count, and corpus size for a given pattern.

Usage
-----
::

    match <pattern> [<model.joblib>]

Arguments
---------
pattern
    Pattern string composed of letters and ``'.'`` wildcards, e.g. ``C.T``
    or ``TH...``.  Case-insensitive.
model
    Path to a joblib model file produced by ``train``
    (default: ``model.joblib``).

Output
------
Four labelled lines are printed::

    Pattern        C.T
    P(match)      0.987654
    E[matches]       12.34
    Words of len       4321

``P(match)`` is the probability that at least one corpus word satisfies the
pattern.  ``E[matches]`` is the expected number of matching words.
``Words of len`` is the total number of corpus words with the same length as
the pattern.
"""
from __future__ import annotations

import argparse

from pattern.adapters.joblib_model_store import JoblibModelStore
from pattern.domain import expected_match_count, match_probability


def main() -> None:
    """Parse arguments, load the model, and print match statistics."""
    parser = argparse.ArgumentParser(
        description="Query a trained pattern-probability model."
    )
    parser.add_argument("pattern", help="Pattern string (letters and '.' for wildcards)")
    parser.add_argument(
        "model",
        nargs="?",
        default="model.joblib",
        help="Path to trained model joblib file (default: model.joblib)",
    )
    args = parser.parse_args()

    model = JoblibModelStore(args.model).load()

    pat = args.pattern.upper()
    p = match_probability(model, pat)
    ex = expected_match_count(model, pat)
    wn = model.word_counts.get(len(pat), 0)

    print(f"Pattern        {pat}")
    print(f"P(match)       {p:.6f}")
    print(f"E[matches]     {ex:.2f}")
    print(f"Words of len   {wn:,}")


if __name__ == "__main__":
    main()
