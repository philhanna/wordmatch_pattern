from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from pattern.domain.model import Model


def build_model(words: Iterable[str], smoothing_k: float = 0.5) -> Model:
    word_counts: dict[int, int] = defaultdict(int)
    pos_freq: dict[int, list[dict[str, int]]] = defaultdict(list)
    bigram_freq: dict[int, list[dict[str, int]]] = defaultdict(list)

    for raw in words:
        word = raw.strip().upper()
        if not word or not word.isalpha():
            continue
        n = len(word)
        word_counts[n] += 1
        while len(pos_freq[n]) < n:
            pos_freq[n].append(defaultdict(int))
        while len(bigram_freq[n]) < n - 1:
            bigram_freq[n].append(defaultdict(int))
        for i, ch in enumerate(word):
            pos_freq[n][i][ch] += 1
            if i < n - 1:
                bigram_freq[n][i][ch + word[i + 1]] += 1

    return Model(
        word_counts=dict(word_counts),
        pos_freq={n: [dict(d) for d in dlist] for n, dlist in pos_freq.items()},
        bigram_freq={n: [dict(d) for d in dlist] for n, dlist in bigram_freq.items()},
        k=smoothing_k,
    )
