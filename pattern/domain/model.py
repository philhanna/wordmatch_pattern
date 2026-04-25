from __future__ import annotations


class Model:
    """Trained model.  All the data needed to answer probability queries."""

    def __init__(
        self,
        word_counts: dict[int, int],
        pos_freq: dict[int, list[dict[str, int]]],
        k: float,
    ) -> None:
        # word_counts[n]  = number of n-letter words in corpus
        self.word_counts = word_counts
        # pos_freq[n][i][c] = number of n-letter words with letter c at position i
        self.pos_freq = pos_freq
        # smoothing constant (add-k)
        self.k = k

    def to_dict(self) -> dict:
        return {
            "word_counts": {str(n): c for n, c in self.word_counts.items()},
            "pos_freq": {str(n): dlist for n, dlist in self.pos_freq.items()},
            "k": self.k,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Model:
        return cls(
            word_counts={int(n): c for n, c in data["word_counts"].items()},
            pos_freq={int(n): dlist for n, dlist in data["pos_freq"].items()},
            k=data["k"],
        )
