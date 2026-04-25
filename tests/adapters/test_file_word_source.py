from pattern.adapters.file_word_source import FileWordSource
from pattern.domain.training import build_model


def test_reads_words(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("cat\ndog\nbird\n")
    words = list(FileWordSource(f).words())
    assert len(words) == 3


def test_accepts_string_path(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("cat\n")
    words = list(FileWordSource(str(f)).words())
    assert len(words) == 1


def test_words_fed_to_build_model(tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("cat\ndog\nbird\n")
    model = build_model(FileWordSource(f).words())
    assert model.word_counts[3] == 2  # cat, dog
    assert model.word_counts[4] == 1  # bird


def test_blank_lines_passed_through(tmp_path):
    # build_model is responsible for skipping blanks; FileWordSource yields them
    f = tmp_path / "words.txt"
    f.write_text("cat\n\ndog\n")
    model = build_model(FileWordSource(f).words())
    assert model.word_counts[3] == 2
