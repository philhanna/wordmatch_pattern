from __future__ import annotations


class Model:
    """Trained model.  All the data needed to answer probability queries.

    The model stores per-length frequency tables computed from a word corpus.
    Given a pattern (letters plus '.' wildcards), these tables let
    :func:`~pattern.domain.inference.match_probability` estimate the
    probability that at least one English word satisfies the pattern.

    Attributes
    ----------
    word_counts : dict[int, int]
        Maps word length *n* to the number of *n*-letter words seen during
        training.
    pos_freq : dict[int, list[dict[str, int]]]
        ``pos_freq[n][i][c]`` is the number of *n*-letter training words that
        have letter *c* at position *i* (0-indexed).
    bigram_freq : dict[int, list[dict[str, int]]]
        ``bigram_freq[n][i]["AB"]`` is the number of *n*-letter training words
        with letter ``'A'`` at position *i* and letter ``'B'`` at position
        *i + 1*.  Covers positions 0 .. *n* - 2.
    k : float
        Laplace (add-*k*) smoothing constant applied during inference to avoid
        zero-probability estimates for unseen letter placements.
    """

    def __init__(
        self,
        word_counts: dict[int, int],
        pos_freq: dict[int, list[dict[str, int]]],
        bigram_freq: dict[int, list[dict[str, int]]],
        k: float,
    ) -> None:
        """Initialise a Model directly from pre-computed frequency tables.

        Prefer :func:`~pattern.domain.training.build_model` over calling this
        constructor directly.

        Parameters
        ----------
        word_counts:
            Number of words seen for each word length.
        pos_freq:
            Per-position letter counts, grouped by word length.
        bigram_freq:
            Adjacent-pair letter counts, grouped by word length.
        k:
            Laplace smoothing constant.
        """
        # word_counts[n]  = number of n-letter words in corpus
        self.word_counts = word_counts
        # pos_freq[n][i][c] = number of n-letter words with letter c at position i
        self.pos_freq = pos_freq
        # bigram_freq[n][i]["AB"] = number of n-letter words with "A" at i, "B" at i+1
        self.bigram_freq = bigram_freq
        # smoothing constant (add-k)
        self.k = k

    def to_dict(self) -> dict:
        """Serialise the model to a JSON-compatible dictionary.

        All integer keys are stringified so the result can be round-tripped
        through :func:`json.dump` / :func:`json.load` without data loss.

        Returns
        -------
        dict
            A plain-Python dictionary suitable for ``json.dump``.
        """
        return {
            "word_counts": {str(n): c for n, c in self.word_counts.items()},
            "pos_freq": {str(n): dlist for n, dlist in self.pos_freq.items()},
            "bigram_freq": {str(n): dlist for n, dlist in self.bigram_freq.items()},
            "k": self.k,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Model:
        """Deserialise a model from the dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data:
            Dictionary as returned by :meth:`to_dict` or loaded from JSON.

        Returns
        -------
        Model
            A fully reconstructed :class:`Model` instance.
        """
        return cls(
            word_counts={int(n): c for n, c in data["word_counts"].items()},
            pos_freq={int(n): dlist for n, dlist in data["pos_freq"].items()},
            # bigram_freq was added after the initial release; tolerate missing key
            bigram_freq={int(n): dlist for n, dlist in data.get("bigram_freq", {}).items()},
            k=data["k"],
        )
