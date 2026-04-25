from pattern.domain.training import build_model

WORDS = ["CAT", "CAR", "CAP", "BAT", "BAR", "BAP", "MAT", "MAR", "MAP"]


def test_word_count():
    model = build_model(WORDS)
    assert model.word_counts[3] == 9


def test_skips_empty_lines():
    model = build_model(["CAT", "", "DOG"])
    assert model.word_counts[3] == 2


def test_skips_non_alpha():
    model = build_model(["CAT", "123", "AB-C", "DOG"])
    assert model.word_counts[3] == 2


def test_normalises_to_uppercase():
    model = build_model(["cat", "Cat", "CAT"])
    assert model.word_counts[3] == 3


def test_position_frequencies():
    model = build_model(WORDS)
    pos0 = model.pos_freq[3][0]
    assert pos0["C"] == 3
    assert pos0["B"] == 3
    assert pos0["M"] == 3


def test_all_words_share_middle_a():
    model = build_model(WORDS)
    assert model.pos_freq[3][1]["A"] == 9


def test_bigram_frequencies():
    model = build_model(WORDS)
    bigram0 = model.bigram_freq[3][0]
    assert bigram0["CA"] == 3
    assert bigram0["BA"] == 3
    assert bigram0["MA"] == 3


def test_smoothing_k_stored():
    model = build_model(WORDS, smoothing_k=1.0)
    assert model.k == 1.0


def test_default_smoothing_k():
    model = build_model(WORDS)
    assert model.k == 0.5


def test_multiple_lengths():
    model = build_model(["CAT", "CATS"])
    assert model.word_counts[3] == 1
    assert model.word_counts[4] == 1
