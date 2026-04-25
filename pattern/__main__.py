import sys
import time

from .domain import match_probability, expected_match_count
from .adapters import FileWordSource
from .application import train

word_file = sys.argv[1] if len(sys.argv) > 1 else "words.txt"

print(f"Training on {word_file} …", end=" ", flush=True)
t0 = time.perf_counter()
model = train(FileWordSource(word_file))
elapsed = time.perf_counter() - t0
total_words = sum(model.word_counts.values())
print(f"done in {elapsed:.2f}s  ({total_words:,} words loaded)\n")

test_patterns = [
    "CAT",
    "C.T",
    "....",
    ".......",
    "TH...",
    "Q...",
    "QZXJW",
    "ZZ...",
    "S....S",
    "PYTHON",
    ".X.",
    "E..E..E",
]

header = f"{'Pattern':<14}  {'P(match)':>10}  {'E[matches]':>12}  {'Words of len':>14}"
print(header)
print("-" * len(header))
for pat in test_patterns:
    p  = match_probability(model, pat)
    ex = expected_match_count(model, pat)
    wn = model.word_counts.get(len(pat), 0)
    print(f"{pat:<14}  {p:>10.4f}  {ex:>12.2f}  {wn:>14,}")
