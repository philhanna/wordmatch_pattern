from typing import Iterable

from pattern.application.train import train
from pattern.ports.word_source import WordSource


class ListWordSource(WordSource):
    def __init__(self, words: list[str]) -> None:
        self._words = words

    def words(self) -> Iterable[str]:
        return iter(self._words)


def test_returns_model_with_correct_word_counts():
    model = train(ListWordSource(["CAT", "DOG", "CAR"]))
    assert model.word_counts[3] == 3


def test_passes_smoothing_k_to_model():
    model = train(ListWordSource(["CAT"]), smoothing_k=2.0)
    assert model.k == 2.0


def test_empty_source_returns_empty_model():
    model = train(ListWordSource([]))
    assert model.word_counts == {}
