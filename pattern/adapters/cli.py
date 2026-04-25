"""
Train a model from a word list and save it as JSON.

Usage
-----
    python -m pattern.adapters.cli <words_file> <output.json> [--smoothing K]
"""
from __future__ import annotations

import argparse
import time

from .file_word_source import FileWordSource
from .json_model_store import JsonModelStore
from ..application import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a pattern-probability model and save it as JSON."
    )
    parser.add_argument("words_file", help="Path to word list (one word per line)")
    parser.add_argument("output", help="Path for the output JSON model file")
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

    store = JsonModelStore(args.output)
    store.save(model)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
