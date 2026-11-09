from __future__ import annotations

import re
from dataclasses import dataclass

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int  # character offsets into the source text
    end: int
    words: int


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _BOUNDARY.split(text) if s.strip()]


def _units(text: str, max_words: int) -> list[tuple[int, int, int]]:
    """Sentence-like units as (start, end, n_words); over-long ones are cut by words."""
    spans, prev = [], 0
    for m in list(_BOUNDARY.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[prev:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((prev + lead, prev + len(seg.rstrip())))
        prev = m.end() if m else len(text)
    units: list[tuple[int, int, int]] = []
    for s, e in spans:
        words = list(_WORD.finditer(text, s, e))
        if len(words) <= max_words:
            units.append((s, e, len(words)))
            continue
        for i in range(0, len(words), max_words):
            part = words[i : i + max_words]
            units.append((part[0].start(), part[-1].end(), len(part)))
    return units


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_words`` words, repeating up to
    ``overlap_words`` words of trailing context so answers that straddle a boundary are
    still retrievable. Chunk text is an exact slice of ``text``."""
    if max_words < 1 or not 0 <= overlap_words < max_words:
        raise ValueError("need max_words >= 1 and 0 <= overlap_words < max_words")
    chunks: list[Chunk] = []
    cur: list[tuple[int, int, int]] = []
    fresh = 0  # units added since the last emitted chunk

    def emit():
        nonlocal fresh
        s, e = cur[0][0], cur[-1][1]
        chunks.append(Chunk(len(chunks), text[s:e], s, e, sum(u[2] for u in cur)))
        fresh = 0

    for u in _units(text, max_words):
        if cur and sum(x[2] for x in cur) + u[2] > max_words:
            emit()
            keep, total = [], 0
            for x in reversed(cur):  # trailing units that fit the overlap budget
                if total + x[2] > overlap_words:
                    break
                keep.insert(0, x)
                total += x[2]
            while keep and total + u[2] > max_words:
                total -= keep.pop(0)[2]
            cur = keep
        cur.append(u)
        fresh += 1
    if cur and fresh:
        emit()
    return chunks
