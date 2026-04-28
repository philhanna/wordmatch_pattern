"""File-based implementation of the :class:`~pattern.ports.WordSource` port.

Reads words line-by-line from a plain-text file.  Non-UTF-8 bytes are silently
replaced rather than raising an error, so the adapter tolerates the occasional
encoding anomaly found in large public word lists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pattern.ports import WordSource


class FileWordSource(WordSource):
    """Adapter: reads words from a flat text file (one word per line).

    Parameters
    ----------
    path:
        Path to the word-list file.  Accepts both :class:`str` and
        :class:`~pathlib.Path`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def words(self) -> Iterable[str]:
        """Yield each line of the file as a raw word string.

        Lines are yielded verbatim (including the trailing newline) so that
        the downstream training layer can strip and validate them uniformly.
        Non-UTF-8 bytes are replaced with the Unicode replacement character
        rather than raising :exc:`UnicodeDecodeError`.

        Yields
        ------
        str
            One raw line per word in the file.

        Raises
        ------
        FileNotFoundError
            If :attr:`path` does not exist.
        """
        with open(self.path, encoding="utf-8", errors="ignore") as fh:
            yield from fh
