import pytest

from pattern.adapters.joblib_model_store import JoblibModelStore
from pattern.domain.training import build_model

WORDS = ["CAT", "DOG", "CAR"]


def test_save_and_load_roundtrip(tmp_path):
    model = build_model(WORDS)
    path = tmp_path / "model.joblib"
    store = JoblibModelStore(path)
    store.save(model)
    loaded = store.load()
    assert loaded.word_counts == model.word_counts
    assert loaded.pos_freq == model.pos_freq
    assert loaded.bigram_freq == model.bigram_freq
    assert loaded.k == model.k


def test_load_missing_file_raises(tmp_path):
    store = JoblibModelStore(tmp_path / "nonexistent.joblib")
    with pytest.raises(FileNotFoundError):
        store.load()


def test_accepts_string_path(tmp_path):
    model = build_model(["CAT"])
    path = tmp_path / "model.joblib"
    JoblibModelStore(str(path)).save(model)
    loaded = JoblibModelStore(str(path)).load()
    assert loaded.word_counts == model.word_counts
