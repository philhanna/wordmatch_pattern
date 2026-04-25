import pytest

from pattern.domain.inference import expected_match_count, match_probability
from pattern.domain.training import build_model

WORDS = ["CAT", "CAR", "CAP", "BAT", "BAR", "BAP", "MAT", "MAR", "MAP"]


@pytest.fixture
def model():
    return build_model(WORDS)


# --- match_probability ---

def test_all_wildcards_is_one(model):
    assert match_probability(model, "...") == 1.0


def test_unknown_length_is_zero(model):
    assert match_probability(model, "XXXX") == 0.0


def test_empty_corpus_is_zero():
    m = build_model([])
    assert match_probability(m, "...") == 0.0


def test_invalid_char_raises(model):
    with pytest.raises(ValueError):
        match_probability(model, "C@T")


def test_digit_in_pattern_raises(model):
    with pytest.raises(ValueError):
        match_probability(model, "C4T")


def test_case_insensitive(model):
    assert match_probability(model, "cat") == match_probability(model, "CAT")


def test_probability_in_unit_interval(model):
    for pat in ["CAT", "C.T", "...", "ZZZ", "XYZ"]:
        p = match_probability(model, pat)
        assert 0.0 <= p <= 1.0, f"P({pat!r}) = {p} out of range"


def test_known_word_beats_impossible_word(model):
    # CAT exists in corpus; ZZZ does not
    assert match_probability(model, "CAT") > match_probability(model, "ZZZ")


def test_wildcard_beats_wrong_letter(model):
    # C.T matches CAT/CAR/CAP; CZT matches nothing
    assert match_probability(model, "C.T") > match_probability(model, "CZT")


# --- expected_match_count ---

def test_all_wildcards_expected_count(model):
    assert expected_match_count(model, "...") == pytest.approx(9.0)


def test_unknown_length_expected_count_is_zero(model):
    assert expected_match_count(model, "XXXX") == 0.0


def test_expected_count_nonnegative(model):
    for pat in ["CAT", "C.T", "...", "ZZZ"]:
        assert expected_match_count(model, pat) >= 0.0


def test_expected_count_scales_with_corpus():
    small = build_model(["CAT", "CAR"])
    large = build_model(["CAT", "CAR", "BAT", "BAR", "MAT", "MAR"])
    assert expected_match_count(large, "...") > expected_match_count(small, "...")
