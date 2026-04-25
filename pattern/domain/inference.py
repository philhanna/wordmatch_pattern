from __future__ import annotations

import math
import re

from .model import Model

_VALID_PATTERN = re.compile(r'^[A-Za-z.]+$')


def match_probability(model: Model, pattern: str) -> float:
    """
    Estimate P(at least one English word matches the pattern).

    Parameters
    ----------
    model   : trained Model from train()
    pattern : string using only letters and '.' (dot = any letter).
              Case-insensitive.

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
        return 0.0

    k = model.k
    pos_data = model.pos_freq.get(n, [])

    log_p_word = 0.0

    for i, ch in enumerate(pattern):
        if ch == '.':
            continue

        freq_dict = pos_data[i] if i < len(pos_data) else {}
        observed = freq_dict.get(ch, 0)
        p_pos = (observed + k) / (W_n + 26 * k)

        if p_pos <= 0.0:
            return 0.0

        log_p_word += math.log(p_pos)

    p_word = min(math.exp(log_p_word), 1.0)

    if p_word >= 1.0:
        return 1.0

    log_prob_none = W_n * math.log1p(-p_word)
    return 1.0 - math.exp(log_prob_none)


def expected_match_count(model: Model, pattern: str) -> float:
    """Return the expected number of matching words (E[X] = W_n * p_word)."""
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
