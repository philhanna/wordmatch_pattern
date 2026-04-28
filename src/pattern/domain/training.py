"""Build a :class:`~pattern.domain.model.Model` from a stream of raw words.

The sole public entry point is :func:`build_model`.  It makes a single pass
over the word stream, accumulating three frequency tables:

* **word_counts** — how many words of each length appear in the corpus.
* **pos_freq** — for each (length, position) pair, how often each letter
  occupies that slot.
* **bigram_freq** — for each (length, position) pair, how often each
  adjacent letter pair *AB* appears at positions (*i*, *i + 1*).

These tables are later consumed by
:func:`~pattern.domain.inference.match_probability` to estimate the
probability that at least one corpus word satisfies a given pattern.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from pattern.domain.model import Model


def build_model(words: Iterable[str], smoothing_k: float = 0.5) -> Model:
    """Train a :class:`~pattern.domain.model.Model` from an iterable of words.

    Each word is upper-cased and stripped of surrounding whitespace before
    being counted.  Words that are empty or contain non-alphabetic characters
    (digits, hyphens, etc.) are silently skipped.

    Parameters
    ----------
    words:
        Any iterable of raw word strings — typically lines from a text file.
        The iterable is consumed exactly once.
    smoothing_k:
        Laplace (add-*k*) smoothing constant stored on the returned model and
        applied during inference.  Larger values pull probabilities toward the
        uniform distribution over 26 letters, which helps on out-of-vocabulary
        letter placements.  Defaults to 0.5 (Jeffreys prior).

    Returns
    -------
    Model
        A trained model ready for queries.  If *words* is empty (or all
        entries are invalid), the returned model contains no frequency data and
        every ``match_probability`` call will return 0.0.
    """
    word_counts: dict[int, int] = defaultdict(int)
    pos_freq: dict[int, list[dict[str, int]]] = defaultdict(list)
    bigram_freq: dict[int, list[dict[str, int]]] = defaultdict(list)

    for raw in words:
        word = raw.strip().upper()
        if not word or not word.isalpha():
            continue
        n = len(word)
        word_counts[n] += 1

        # Extend frequency lists to cover all positions in this word
        while len(pos_freq[n]) < n:
            pos_freq[n].append(defaultdict(int))
        while len(bigram_freq[n]) < n - 1:
            bigram_freq[n].append(defaultdict(int))

        for i, ch in enumerate(word):
            pos_freq[n][i][ch] += 1
            if i < n - 1:
                # Record the ordered pair (ch, next_ch) starting at position i
                bigram_freq[n][i][ch + word[i + 1]] += 1

    return Model(
        word_counts=dict(word_counts),
        pos_freq={n: [dict(d) for d in dlist] for n, dlist in pos_freq.items()},
        bigram_freq={n: [dict(d) for d in dlist] for n, dlist in bigram_freq.items()},
        k=smoothing_k,
    )
