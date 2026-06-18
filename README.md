# pattern

Estimate the probability that **at least one English word** matches a given
letter pattern such as `C.T`, `TH...`, or `E..E..E` — where `.` is a wildcard
that stands for any single letter.

Given a word list, `pattern` trains a lightweight statistical model from
per-position letter frequencies and adjacent-letter (bigram) frequencies. It
can then answer two questions for any pattern:

- **P(match)** — the probability that *some* word in the corpus satisfies the
  pattern.
- **E[matches]** — the expected *number* of words that satisfy the pattern.

This is handy for crossword tooling, word-game heuristics, and any situation
where you want a cheap estimate of how "fillable" a slot is without scanning
the entire dictionary at query time.

---

## How it works

Training makes a single pass over the corpus and builds three frequency tables,
keyed by word length *n*:

| Table          | Meaning                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| `word_counts`  | How many *n*-letter words were seen.                                    |
| `pos_freq`     | `pos_freq[n][i][c]` — how often letter `c` appears at position `i`.     |
| `bigram_freq`  | `bigram_freq[n][i]["AB"]` — how often pair `AB` appears at `(i, i+1)`.  |

At query time, for an *n*-letter pattern the per-word match probability `p` is
the product of factors from the constrained positions:

- **Isolated constrained letter** — uses the position table with add-*k*
  (Laplace) smoothing:

  ```
  P(c at i) = (count(c, i) + k) / (W_n + 26k)
  ```

- **Adjacent constrained letters** — the second letter is modelled
  conditionally on the first via the bigram table, so common digraphs like
  `TH` and `QU` are not unfairly penalised:

  ```
  P(B at i+1 | A at i) = (count("AB", i) + k) / (count(A, i) + 26k)
  ```

- Wildcards (`.`) contribute a factor of 1 and break any adjacent-pair chain.

Treating each of the `W_n` corpus words as an independent Bernoulli trial, the
probability that **at least one** matches is the complement of "none match":

```
P(match) = 1 - (1 - p)^W_n
```

and the expected count is simply `E[matches] = W_n · p` (linearity of
expectation). All products are accumulated in log-space for numerical
stability.

The smoothing constant *k* defaults to `0.5` (the Jeffreys prior). Larger
values pull estimates toward the uniform distribution over 26 letters, which
helps when the corpus is small or sparse.

---

## Architecture

The project follows a **hexagonal (ports & adapters)** design. Dependencies
point inward: the domain knows nothing about files or JSON.

```
╔════════════════════════════════════════════════════════════════════╗
║  Driving Side (Inbound)                                            ║
║  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────┐ ║
║  │  train_cli.main()  │  │  match_cli.main()  │  │  __main__.py  │ ║
║  │  (train command)   │  │  (match command)   │  │  (smoke test) │ ║
║  └─────────┬──────────┘  └─────────┬──────────┘  └───────┬───────┘ ║
╚════════════│═══════════════════════│═════════════════════│═════════╝
             ▼                       ▼                      ▼
╔════════════════════════════════════════════════════════════════════╗
║  Application Layer                                                 ║
║    train(source: WordSource) -> Model                              ║
║    load_model(path) -> Model                                       ║
║    depends on ports ─────────────┐                                 ║
╚══════════════════════════════════│═════════════════════════════════╝
                                    ▼
╔════════════════════════════════════════════════════════════════════╗
║  Ports (abstract interfaces)                                       ║
║    WordSource.words() -> Iterable[str]                             ║
║    ModelStore.save(model) / .load() -> Model                       ║
╚════════════════════════════════════════════════════════════════════╝
             ▲                                       ▲
╔════════════│═══════════════════════════════════════│════════════════╗
║  Driven Side (Outbound Adapters)                   │                ║
║  ┌──────────┴───────────┐          ┌───────────────┴─────────────┐  ║
║  │   FileWordSource     │          │      JsonModelStore         │  ║
║  │ (reads words file)   │          │ (saves/loads model as JSON) │  ║
║  └──────────────────────┘          └─────────────────────────────┘  ║
╚═════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔════════════════════════════════════════════════════════════════════╗
║  Domain Layer (pure, no I/O)                                       ║
║    Model               — frequency tables + smoothing constant     ║
║    build_model(words)  — training                                  ║
║    match_probability() / expected_match_count() — inference        ║
╚════════════════════════════════════════════════════════════════════╝
```

**Data flow:** CLI → `train(WordSource)` → `build_model` → `Model` →
`JsonModelStore.save` → *(later)* `JsonModelStore.load` → `match_probability`.

**Dependency rule:** the domain (`pattern.domain`) imports nothing outside
itself. The application layer depends only on ports; adapters implement the
ports; the CLI wires concrete adapters to the application use cases.

### Package layout

```
pattern/
├── domain/           # Pure logic — no I/O
│   ├── model.py        Model (frequency tables, to_dict/from_dict)
│   ├── training.py     build_model()
│   └── inference.py    match_probability(), expected_match_count()
├── ports/            # Abstract interfaces
│   ├── word_source.py  WordSource
│   └── model_store.py  ModelStore
├── adapters/         # Concrete I/O implementations
│   ├── file_word_source.py   FileWordSource
│   ├── json_model_store.py   JsonModelStore
│   ├── train_cli.py          `train` command
│   └── match_cli.py          `match` command
├── application/      # Use-case orchestration
│   ├── train.py        train()
│   └── query.py        load_model()
└── __main__.py       # `python -m pattern` smoke test
```

---

## Installation

Requires **Python ≥ 3.12**.

```bash
# From the project root
pip install -e ".[dev]"     # editable install with pytest for development
# or
pip install .               # plain install
```

Installing registers two console scripts, `train` and `match` (see
[`pyproject.toml`](pyproject.toml)).

---

## Usage

### 1. Train a model

```bash
train <words_file> <output.json> [--smoothing K]
```

`words_file` is a plain-text file with one word per line. Words are
upper-cased and stripped; entries that are empty or contain non-alphabetic
characters are silently skipped.

```bash
train words.txt model.json
train /usr/share/dict/words model.json --smoothing 1.0
```

This repository ships a `words.txt` corpus (~345k entries) to get started.

### 2. Query a pattern

```bash
match <pattern> [<model.json>]
```

`pattern` uses letters and `.` wildcards (case-insensitive). The model path
defaults to `model.json`.

```bash
$ match C.T model.json
Pattern        C.T
P(match)       0.999998
E[matches]     42.17
Words of len   8,124
```

### 3. Quick smoke test

Train in-memory and print a table of representative patterns in one shot,
without saving a model:

```bash
python -m pattern                 # uses words.txt by default
python -m pattern /usr/share/dict/words
```

### Convenience wrappers

The repo also includes two thin shell scripts, [`train`](train) and
[`match`](match), that simply `exec` the corresponding CLI module — useful if
you have not installed the package:

```bash
./train words.txt model.json
./match TH... model.json
```

---

## Library API

The full public API is re-exported from the top-level package:

```python
from pattern import (
    train, load_model,
    match_probability, expected_match_count,
    Model, FileWordSource, JsonModelStore,
)

# Train from any WordSource (here, a file)
model = train(FileWordSource("words.txt"), smoothing_k=0.5)

# Persist / reload
JsonModelStore("model.json").save(model)
model = load_model("model.json")        # or JsonModelStore("model.json").load()

# Query
match_probability(model, "C.T")         # -> float in [0.0, 1.0]
expected_match_count(model, "C.T")      # -> float >= 0.0
```

Because `train()` accepts any `WordSource`, you can feed words from anywhere by
implementing the port:

```python
from pattern import WordSource, train

class ListWordSource(WordSource):
    def __init__(self, words): self._words = words
    def words(self): return iter(self._words)

model = train(ListWordSource(["cat", "cot", "cut", "cab"]))
```

Likewise, implement `ModelStore` to persist models to a database, S3, or any
other backend without touching the rest of the code.

---

## Model file format

`JsonModelStore` writes a compact (`separators=(",", ":")`) JSON document.
Integer keys are stringified so the structure round-trips cleanly through
`json`:

```json
{
  "word_counts": {"3": 8124, "4": 12950},
  "pos_freq":    {"3": [{"C": 412, ...}, ...]},
  "bigram_freq": {"3": [{"CA": 88, ...}, ...]},
  "k": 0.5
}
```

`Model.from_dict` tolerates a missing `bigram_freq` key, so models produced by
earlier versions still load.

---

## Testing

Tests are written with **pytest** and mirror the package structure under
[`tests/`](tests/):

```bash
pytest
```

`testpaths` is pre-configured in [`pyproject.toml`](pyproject.toml), so a bare
`pytest` from the project root discovers everything in `tests/`.

---

## Project layout reference

| Path                              | Purpose                                  |
| --------------------------------- | ---------------------------------------- |
| [`pattern/`](pattern/)            | Library source (hexagonal layers).       |
| [`tests/`](tests/)                | Pytest suite, one subpackage per layer.  |
| [`words.txt`](words.txt)          | Sample word-list corpus.                 |
| [`train`](train) / [`match`](match) | Shell wrappers around the CLI modules. |
| [`pyproject.toml`](pyproject.toml) | Build config, scripts, pytest settings. |
</content>
</invoke>
