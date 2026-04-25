"""
pattern_prob.py
===============
Train a model from a word list, then estimate the probability that a given
regex pattern (letters and '.' only) matches at least one English word.

Usage
-----
    from pattern_prob import train, match_probability

    model = train("words.txt")
    p = match_probability(model, "C.T")      # → ~0.99
    p = match_probability(model, "QZ.ZQ")   # → ~0.0
    p = match_probability(model, "......")   # → ~1.0 (6-letter words abound)

Model overview
--------------
For a pattern of length n with fixed letters at some positions:

1.  Count W_n = total words of length n in the corpus.

2.  For each fixed position i with letter c, look up f(i, c, n) = fraction of
    n-letter words that have letter c at position i.  These are estimated from
    the corpus and stored in the model.

3.  Treat constraints as *approximately independent* across positions (a known
    simplification — English has correlations, but independence is a reasonable
    first approximation).  Under independence:

        P(a random n-letter word matches) ≈ ∏  f(i, c_i, n)
                                            fixed i

4.  If there are W_n words of length n, and each independently matches with
    probability p_word, then the number of matches is Binomial(W_n, p_word).
    P(at least one match) = 1 − (1 − p_word)^W_n.

Smoothing
---------
Zero observations at a position/letter/length cell don't mean impossible —
they just mean rare.  We apply add-k (Laplace) smoothing with k=0.5 so that
unseen letter/position combos get a small but nonzero probability.
"""

from __future__ import annotations

import re
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


# -- Data structure ------------------------------------------------------------

class Model(NamedTuple):
    """Trained model.  All the data needed to answer probability queries."""
    # word_counts[n]  = number of n-letter words in corpus
    word_counts: dict[int, int]
    # pos_freq[n][i][c] = number of n-letter words with letter c at position i
    pos_freq: dict[int, list[dict[str, int]]]
    # smoothing constant (add-k)
    k: float


# -- Training ------------------------------------------------------------------

def train(word_file: str | Path, smoothing_k: float = 0.5) -> Model:
    """
    Read a word list (one word per line, any case) and build the model.

    Parameters
    ----------
    word_file   : path to the flat word file
    smoothing_k : Laplace smoothing constant (default 0.5, i.e. Jeffreys prior)

    Returns
    -------
    A trained Model ready for use with match_probability().
    """
    word_counts: dict[int, int] = defaultdict(int)
    # pos_freq[n] is a list of dicts, one per position
    pos_freq: dict[int, list[dict[str, int]]] = defaultdict(list)

    with open(word_file, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            word = raw.strip().upper()
            if not word or not word.isalpha():
                continue
            n = len(word)
            word_counts[n] += 1
            # Extend the position list if this is the first word of length n
            # or a longer word at an already-seen length (shouldn't happen, but safe)
            while len(pos_freq[n]) < n:
                pos_freq[n].append(defaultdict(int))
            for i, ch in enumerate(word):
                pos_freq[n][i][ch] += 1

    return Model(
        word_counts=dict(word_counts),
        pos_freq={n: [dict(d) for d in dlist] for n, dlist in pos_freq.items()},
        k=smoothing_k,
    )


# -- Inference -----------------------------------------------------------------

_VALID_PATTERN = re.compile(r'^[A-Za-z.]+$')

def match_probability(model: Model, pattern: str) -> float:
    """
    Estimate P(at least one English word matches the pattern).

    Parameters
    ----------
    model   : trained Model from train()
    pattern : regex string using only letters and '.' (dot = any letter)
              Length must be 1–30.  Case-insensitive.

    Returns
    -------
    Float in [0.0, 1.0].

    Raises
    ------
    ValueError if the pattern contains characters other than letters and '.'.
    """
    pattern = pattern.upper()

    if not _VALID_PATTERN.match(pattern):
        raise ValueError(
            f"Pattern must contain only letters and '.', got: {pattern!r}"
        )

    n = len(pattern)
    W_n = model.word_counts.get(n, 0)

    if W_n == 0:
        # No words of this length in the corpus at all
        return 0.0

    k = model.k
    pos_data = model.pos_freq.get(n, [])

    # -- Compute log P(a random n-letter word matches) -------------------------
    # Under independence across positions, this is the product of per-position
    # probabilities.  Work in log-space to avoid underflow.

    log_p_word = 0.0  # log(1.0) — start at probability 1

    for i, ch in enumerate(pattern):
        if ch == '.':
            continue  # wildcard: any letter is fine, contributes factor 1

        # Smoothed frequency of letter ch at position i among n-letter words
        freq_dict = pos_data[i] if i < len(pos_data) else {}
        observed  = freq_dict.get(ch, 0)
        # Denominator: total n-letter words + 26*k  (one smoothing slot per letter)
        p_pos = (observed + k) / (W_n + 26 * k)

        if p_pos <= 0.0:
            # Should not happen with smoothing > 0, but be safe
            return 0.0

        log_p_word += math.log(p_pos)

    p_word = min(math.exp(log_p_word), 1.0)  # clamp: smoothing can't exceed 1

    # -- P(at least one match) = 1 − (1 − p_word)^W_n ------------------------
    if p_word >= 1.0:
        return 1.0
    # Use log1p for numerical stability when p_word is tiny
    log_prob_none = W_n * math.log1p(-p_word)
    p_at_least_one = 1.0 - math.exp(log_prob_none)

    return p_at_least_one


# -- Convenience: top matches & diagnostics ------------------------------------

def expected_match_count(model: Model, pattern: str) -> float:
    """Return the expected *number* of matching words (E[X] = W_n * p_word)."""
    pattern = pattern.upper()
    n = len(pattern)
    W_n = model.word_counts.get(n, 0)
    if W_n == 0:
        return 0.0
    k = model.k
    pos_data = model.pos_freq.get(n, [])
    log_p = 0.0
    for i, ch in enumerate(pattern):
        if ch == '.':
            continue
        freq_dict = pos_data[i] if i < len(pos_data) else {}
        observed = freq_dict.get(ch, 0)
        log_p += math.log((observed + k) / (W_n + 26 * k))
    return W_n * min(math.exp(log_p), 1.0)


# -- CLI / demo ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    word_file = sys.argv[1] if len(sys.argv) > 1 else "words.txt"

    print(f"Training on {word_file} …", end=" ", flush=True)
    t0 = time.perf_counter()
    model = train(word_file)
    elapsed = time.perf_counter() - t0
    total_words = sum(model.word_counts.values())
    print(f"done in {elapsed:.2f}s  ({total_words:,} words loaded)\n")

    test_patterns = [
        "CAT",          # exact word — should be ~1.0
        "C.T",          # many matches (CAT, CUT, COT, …)
        "....",         # all 4-letter words
        ".......",      # all 7-letter words
        "TH...",        # common English prefix
        "Q...",         # Q-words are rare
        "QZXJW",        # near-impossible consonant cluster
        "ZZ...",        # ZZ start — very rare
        "S....S",       # plural-ish pattern
        "PYTHON",       # proper noun / brand — may or may not be in list
        ".X.",          # X in the middle
        "E..E..E",      # vowel frame
    ]

    header = f"{'Pattern':<14}  {'P(match)':>10}  {'E[matches]':>12}  {'Words of len':>14}"
    print(header)
    print("-" * len(header))
    for pat in test_patterns:
        p   = match_probability(model, pat)
        ex  = expected_match_count(model, pat)
        wn  = model.word_counts.get(len(pat), 0)
        print(f"{pat:<14}  {p:>10.4f}  {ex:>12.2f}  {wn:>14,}")
