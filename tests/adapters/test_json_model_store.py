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


def test_save_uses_compact_json(tmp_path):
    model = build_model(["CAT"])
    path = tmp_path / "model.json"
    JsonModelStore(path).save(model)
    assert path.read_text(encoding="utf-8") == (
        '{"word_counts":{"3":1},"pos_freq":{"3":[{"C":1},{"A":1},{"T":1}]},'
        '"bigram_freq":{"3":[{"CA":1},{"AT":1}]},"k":0.5}'
    )
