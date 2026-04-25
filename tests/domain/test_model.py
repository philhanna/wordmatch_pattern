from pattern.domain.model import Model
from pattern.domain.training import build_model

WORDS = ["CAT", "CAR", "CAP"]


def test_roundtrip():
    original = build_model(WORDS)
    restored = Model.from_dict(original.to_dict())
    assert restored.word_counts == original.word_counts
    assert restored.pos_freq == original.pos_freq
    assert restored.bigram_freq == original.bigram_freq
    assert restored.k == original.k


def test_to_dict_stringifies_length_keys():
    model = build_model(WORDS)
    d = model.to_dict()
    assert all(isinstance(k, str) for k in d["word_counts"])
    assert all(isinstance(k, str) for k in d["pos_freq"])


def test_from_dict_restores_int_keys():
    model = build_model(WORDS)
    restored = Model.from_dict(model.to_dict())
    assert all(isinstance(k, int) for k in restored.word_counts)
    assert all(isinstance(k, int) for k in restored.pos_freq)


def test_empty_corpus_roundtrip():
    original = build_model([])
    restored = Model.from_dict(original.to_dict())
    assert restored.word_counts == {}
    assert restored.k == original.k
