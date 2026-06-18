"""Command-line interface for training a pattern-probability model.

Reads a word list from a text file, builds a
:class:`~pattern.domain.model.Model`, and saves it as a joblib file that can
later be queried with the ``match`` command.

Usage
-----
::

    train <words_file> <output.joblib> [--smoothing K]

Arguments
---------
words_file
    Path to a plain-text file containing one word per line.
output
    Destination path for the trained model (joblib format).
--smoothing K
    Laplace smoothing constant *k* (default: 0.5).  Increase to pull
    probability estimates toward the uniform distribution when the corpus is
    small or sparse.

Examples
--------
::

    train /usr/share/dict/words model.joblib
    train words.txt model.joblib --smoothing 1.0
"""
from __future__ import annotations

import argparse
import time

from pattern.adapters.file_word_source import FileWordSource
from pattern.adapters.joblib_model_store import JoblibModelStore
from pattern.application import train


def main() -> None:
    """Parse arguments, train the model, and save it to disk."""
    parser = argparse.ArgumentParser(
        description="Train a pattern-probability model and save it with joblib."
    )
    parser.add_argument("words_file", help="Path to word list (one word per line)")
    parser.add_argument("output", nargs="?", default="model.joblib", help="Path for the output joblib model file")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        metavar="K",
        help="Laplace smoothing constant (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"Training on {args.words_file} …", end=" ", flush=True)
    t0 = time.perf_counter()
    model = train(FileWordSource(args.words_file), smoothing_k=args.smoothing)
    elapsed = time.perf_counter() - t0
    total_words = sum(model.word_counts.values())
    print(f"done in {elapsed:.2f}s  ({total_words:,} words loaded)")

    store = JoblibModelStore(args.output)
    store.save(model)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
