from pattern.domain.model import Model
from pattern.domain.training import build_model

WORDS = ["CAT", "CAR", "CAP"]


def test_constructor_stores_attributes():
    model = Model(
        word_counts={3: 2},
        pos_freq={3: [{"C": 2}, {"A": 2}, {"T": 1, "R": 1}]},
        bigram_freq={3: [{"CA": 2}, {"AT": 1, "AR": 1}]},
        k=0.5,
    )
    assert model.word_counts == {3: 2}
    assert model.pos_freq == {3: [{"C": 2}, {"A": 2}, {"T": 1, "R": 1}]}
    assert model.bigram_freq == {3: [{"CA": 2}, {"AT": 1, "AR": 1}]}
    assert model.k == 0.5


def test_build_model_uses_int_length_keys():
    model = build_model(WORDS)
    assert all(isinstance(k, int) for k in model.word_counts)
    assert all(isinstance(k, int) for k in model.pos_freq)
    assert all(isinstance(k, int) for k in model.bigram_freq)


def test_empty_corpus_has_no_counts():
    model = build_model([])
    assert model.word_counts == {}
    assert model.pos_freq == {}
    assert model.bigram_freq == {}
    assert model.k == 0.5
