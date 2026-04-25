from pattern.adapters.json_model_store import JsonModelStore
from pattern.application.query import load_model
from pattern.domain.training import build_model


def test_load_model_returns_correct_model(tmp_path):
    model = build_model(["CAT", "DOG"])
    path = tmp_path / "model.json"
    JsonModelStore(path).save(model)
    loaded = load_model(path)
    assert loaded.word_counts == model.word_counts
    assert loaded.k == model.k


def test_load_model_accepts_string_path(tmp_path):
    model = build_model(["CAT"])
    path = tmp_path / "model.json"
    JsonModelStore(path).save(model)
    loaded = load_model(str(path))
    assert loaded.word_counts == model.word_counts
