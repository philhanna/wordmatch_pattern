"""Pattern-matching probability and expected-count inference.

Both public functions operate on a trained :class:`~pattern.domain.model.Model`
and a *pattern* string composed of letters and ``'.'`` wildcards.

Probability model
-----------------
For a given *n*-letter pattern with *k* constrained positions, the inference
works as follows:

1. **Independent positions** — Each constrained letter that does not form an
   adjacent pair with another constrained letter contributes a factor::

       P(letter c at position i) = (count(c, i) + k) / (W_n + 26*k)

   where *W_n* is the total number of *n*-letter corpus words and *k* is the
   Laplace smoothing constant.

2. **Adjacent constrained pairs** — When two consecutive positions are both
   constrained, the second is modelled conditionally on the first using the
   bigram table::

       P(B at i+1 | A at i) = (count("AB", i) + k) / (count(A, i) + 26*k)

   This avoids over-penalising common digraphs (e.g. ``TH``, ``QU``) that
   appear far more often together than independence would predict.

3. **At-least-one probability** — The per-word probability *p* computed above
   assumes each corpus word is an independent Bernoulli trial.  The
   probability that *at least one* of the *W_n* words matches is::

       P(match) = 1 - (1 - p)^W_n
"""
from __future__ import annotations

import math
import re

from pattern.domain.model import Model

_VALID_PATTERN = re.compile(r'^[A-Za-z.]+$')


def match_probability(model: Model, pattern: str) -> float:
    """Estimate P(at least one English word matches the pattern).

    Parameters
    ----------
    model:
        Trained :class:`~pattern.domain.model.Model` from
        :func:`~pattern.domain.training.build_model`.
    pattern:
        String using only letters and ``'.'`` (dot = any letter).
        Case-insensitive.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If *pattern* contains characters other than letters and ``'.'``.
    """
    pattern = pattern.upper()

    if not _VALID_PATTERN.match(pattern):
        raise ValueError(
            f"Pattern must contain only letters and '.', got: {pattern!r}"
        )

    n = len(pattern)
    W_n = model.word_counts.get(n, 0)

    # No words of this length in the corpus — no match is possible
    if W_n == 0:
        return 0.0

    k = model.k
    pos_data = model.pos_freq.get(n, [])
    bigram_data = model.bigram_freq.get(n, [])

    log_p_word = 0.0  # accumulates log P(a single random word matches)
    prev_pos = -2     # position of the last constrained letter (-2 = none)
    prev_ch = ''      # that letter itself

    for i, ch in enumerate(pattern):
        if ch == '.':
            # Wildcard: contributes factor 1 regardless of the actual letter;
            # also breaks any adjacent-pair chain for subsequent positions.
            prev_pos = -2
            prev_ch = ''
            continue

        if i == prev_pos + 1:
            # Two consecutive constrained letters: use the bigram table so
            # that common digraphs (e.g. "TH") are not unfairly penalised.
            bigram_dict = bigram_data[i - 1] if i - 1 < len(bigram_data) else {}
            pair_count = bigram_dict.get(prev_ch + ch, 0)
            unigram_dict = pos_data[i - 1] if i - 1 < len(pos_data) else {}
            prev_count = unigram_dict.get(prev_ch, 0)
            # Conditional probability: P(ch | prev_ch) at this position
            p = (pair_count + k) / (prev_count + 26 * k)
        else:
            # Non-adjacent constrained letter: use the unigram (position) table
            freq_dict = pos_data[i] if i < len(pos_data) else {}
            observed = freq_dict.get(ch, 0)
            # Marginal probability: P(ch at position i)
            p = (observed + k) / (W_n + 26 * k)

        if p <= 0.0:
            return 0.0

        log_p_word += math.log(p)
        prev_pos = i
        prev_ch = ch

    # p_word is the probability that one randomly chosen n-letter word matches
    p_word = min(math.exp(log_p_word), 1.0)

    if p_word >= 1.0:
        return 1.0

    # At-least-one probability via the complement of "no word matches"
    log_prob_none = W_n * math.log1p(-p_word)
    return 1.0 - math.exp(log_prob_none)


def expected_match_count(model: Model, pattern: str) -> float:
    """Return the expected number of corpus words that match *pattern*.

    By linearity of expectation, this equals *W_n* × *p_word*, where *p_word*
    is the probability that a single randomly chosen *n*-letter corpus word
    matches.

    Parameters
    ----------
    model:
        Trained :class:`~pattern.domain.model.Model`.
    pattern:
        Pattern string (letters and ``'.'``).  Case-insensitive.

    Returns
    -------
    float
        Expected match count, ≥ 0.
    """
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
