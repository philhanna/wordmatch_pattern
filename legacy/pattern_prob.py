"""
pattern_prob.py — Positional unigram model for crossword pattern matching.

Estimates the probability that a given pattern (letters + '.' wildcards) matches
at least one word in the trained lexicon.

Usage:
    model = train("words.txt")          # words.txt: one UPPERCASE word per line
    p = match_probability(model, "C.T") # e.g. 0.9997...
    n = expected_match_count(model, "C.T")

How it works:
  1. Per-position letter frequencies (with Laplace smoothing) give P(word matches).
  2. Naive Bayes independence assumption: multiply per-position probabilities.
  3. Binomial complement: P(at least one match) = 1 − (1 − p_word)^W_n
"""

import math
import re
from collections import defaultdict


# ── Training ──────────────────────────────────────────────────────────────────

def train(word_file: str) -> dict:
    """
    Read *word_file* (one uppercase word per line) and return a model dict:
        {
          "counts":  {length: {pos: {letter: count}}},
          "totals":  {length: word_count},
          "k":       Laplace smoothing constant (0.5),
          "vocab":   26  (fixed alphabet size),
        }
    """
    counts  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals  = defaultdict(int)

    with open(word_file, "r") as fh:
        for raw in fh:
            word = raw.strip().upper()
            if not word or not word.isalpha():
                continue
            n = len(word)
            totals[n] += 1
            for i, ch in enumerate(word):
                counts[n][i][ch] += 1

    return {
        "counts": {n: {i: dict(pos) for i, pos in pos_map.items()}
                   for n, pos_map in counts.items()},
        "totals": dict(totals),
        "k":      0.5,
        "vocab":  26,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def match_probability(model: dict, pattern: str) -> float:
    """
    Return P(pattern matches ≥ 1 word in the lexicon).

    *pattern* consists of uppercase letters and '.' wildcards only.
    Returns 0.0 immediately for lengths not seen during training.
    """
    pattern = pattern.upper()
    if not re.fullmatch(r"[A-Z.]+", pattern):
        raise ValueError(f"Pattern must contain only A-Z and '.': {pattern!r}")

    n      = len(pattern)
    counts = model["counts"]
    totals = model["totals"]
    k      = model["k"]
    V      = model["vocab"]

    W_n = totals.get(n, 0)
    if W_n == 0:
        return 0.0

    # ── Per-word match probability (log space) ────────────────────────────────
    log_p_word = 0.0
    pos_counts = counts.get(n, {})

    for i, ch in enumerate(pattern):
        if ch == ".":
            continue  # wildcard: factor = 1, log = 0
        pos_freq = pos_counts.get(i, {})
        raw_count = pos_freq.get(ch, 0)
        # Laplace-smoothed: (count + k) / (W_n + k*V)
        p_pos = (raw_count + k) / (W_n + k * V)
        log_p_word += math.log(p_pos)

    p_word = min(math.exp(log_p_word), 1.0)   # clamp: smoothing can overshoot

    # ── P(at least one match) = 1 − (1 − p_word)^W_n ────────────────────────
    if p_word >= 1.0:
        return 1.0
    log_prob_none   = W_n * math.log1p(-p_word)   # numerically stable
    p_at_least_one  = 1.0 - math.exp(log_prob_none)
    return p_at_least_one


def expected_match_count(model: dict, pattern: str) -> float:
    """
    Return the expected number of words in the lexicon that match *pattern*.
    Useful for debugging / calibration.
    """
    pattern = pattern.upper()
    if not re.fullmatch(r"[A-Z.]+", pattern):
        raise ValueError(f"Pattern must contain only A-Z and '.': {pattern!r}")

    n      = len(pattern)
    counts = model["counts"]
    totals = model["totals"]
    k      = model["k"]
    V      = model["vocab"]

    W_n = totals.get(n, 0)
    if W_n == 0:
        return 0.0

    log_p = 0.0
    pos_counts = counts.get(n, {})
    for i, ch in enumerate(pattern):
        if ch == ".":
            continue
        pos_freq  = pos_counts.get(i, {})
        raw_count = pos_freq.get(ch, 0)
        p_pos     = (raw_count + k) / (W_n + k * V)
        log_p    += math.log(p_pos)

    return W_n * min(math.exp(log_p), 1.0)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pattern_prob.py words.txt [PATTERN ...]")
        sys.exit(1)

    word_file = sys.argv[1]
    patterns  = sys.argv[2:] or ["C.T", "PY..ON", "....", "ZZZZ"]

    print(f"Training on {word_file} …", end=" ", flush=True)
    model = train(word_file)
    print("done.")
    print(f"  Word lengths in model: {sorted(model['totals'].keys())}")
    print()

    for pat in patterns:
        p = match_probability(model, pat)
        n = expected_match_count(model, pat)
        print(f"  {pat:20s}  P={p:.6f}   E[matches]={n:.1f}")
