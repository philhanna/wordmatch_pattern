"""
Query a trained model for a pattern.

Usage
-----
    python -m pattern.adapters.match_cli <pattern> <model.json>
"""
from __future__ import annotations

import argparse

from pattern.adapters.json_model_store import JsonModelStore
from pattern.domain import expected_match_count, match_probability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query a trained pattern-probability model."
    )
    parser.add_argument("pattern", help="Pattern string (letters and '.' for wildcards)")
    parser.add_argument("model", help="Path to trained model JSON file")
    args = parser.parse_args()

    model = JsonModelStore(args.model).load()

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
