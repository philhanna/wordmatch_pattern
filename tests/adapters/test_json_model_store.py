import pytest

from pattern.adapters.json_model_store import JsonModelStore
from pattern.domain.training import build_model

WORDS = ["CAT", "DOG", "CAR"]


def test_save_and_load_roundtrip(tmp_path):
    model = build_model(WORDS)
    path = tmp_path / "model.json"
    store = JsonModelStore(path)
    store.save(model)
    loaded = store.load()
    assert loaded.word_counts == model.word_counts
    assert loaded.pos_freq == model.pos_freq
    assert loaded.bigram_freq == model.bigram_freq
    assert loaded.k == model.k


def test_load_missing_file_raises(tmp_path):
    store = JsonModelStore(tmp_path / "nonexistent.json")
    with pytest.raises(FileNotFoundError):
        store.load()


def test_accepts_string_path(tmp_path):
    model = build_model(["CAT"])
    path = tmp_path / "model.json"
    JsonModelStore(str(path)).save(model)
    loaded = JsonModelStore(str(path)).load()
    assert loaded.word_counts == model.word_counts
