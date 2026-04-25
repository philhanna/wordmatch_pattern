from __future__ import annotations

import math
import re

from pattern.domain.model import Model

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

    bigram_data = model.bigram_freq.get(n, [])
    log_p_word = 0.0
    prev_pos = -2
    prev_ch = ''

    for i, ch in enumerate(pattern):
        if ch == '.':
            prev_pos = -2
            prev_ch = ''
            continue

        if i == prev_pos + 1:
            # Adjacent constrained pair: use P(ch | prev_ch) from bigram table
            bigram_dict = bigram_data[i - 1] if i - 1 < len(bigram_data) else {}
            pair_count = bigram_dict.get(prev_ch + ch, 0)
            unigram_dict = pos_data[i - 1] if i - 1 < len(pos_data) else {}
            prev_count = unigram_dict.get(prev_ch, 0)
            p = (pair_count + k) / (prev_count + 26 * k)
        else:
            freq_dict = pos_data[i] if i < len(pos_data) else {}
            observed = freq_dict.get(ch, 0)
            p = (observed + k) / (W_n + 26 * k)

        if p <= 0.0:
            return 0.0

        log_p_word += math.log(p)
        prev_pos = i
        prev_ch = ch

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
    bigram_data = model.bigram_freq.get(n, [])
    log_p = 0.0
    prev_pos = -2
    prev_ch = ''
    for i, ch in enumerate(pattern):
        if ch == '.':
            prev_pos = -2
            prev_ch = ''
            continue
        if i == prev_pos + 1:
            bigram_dict = bigram_data[i - 1] if i - 1 < len(bigram_data) else {}
            pair_count = bigram_dict.get(prev_ch + ch, 0)
            unigram_dict = pos_data[i - 1] if i - 1 < len(pos_data) else {}
            prev_count = unigram_dict.get(prev_ch, 0)
            log_p += math.log((pair_count + k) / (prev_count + 26 * k))
        else:
            freq_dict = pos_data[i] if i < len(pos_data) else {}
            observed = freq_dict.get(ch, 0)
            log_p += math.log((observed + k) / (W_n + 26 * k))
        prev_pos = i
        prev_ch = ch
    return W_n * min(math.exp(log_p), 1.0)
