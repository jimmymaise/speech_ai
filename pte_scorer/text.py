"""Text normalisation and tokenisation of the reference passage / ASR output."""
from __future__ import annotations

import re
from dataclasses import dataclass

from num2words import num2words

FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "ah", "eh", "uhm", "umm"}

_PUNCT_BREAK = re.compile(r"[,.;:!?—–-]$")


@dataclass
class RefWord:
    text: str            # normalised word used for matching
    raw: str             # word as shown in the passage
    break_after: bool    # punctuation after it -> a pause here is acceptable


def _expand_numbers(s: str) -> str:
    def pct(m):
        return num2words(int(m.group(1))) + " percent"

    def year_or_num(m):
        n = int(m.group(0))
        if 1100 <= n <= 2099:
            return num2words(n, to="year")
        return num2words(n)

    s = re.sub(r"(\d+)\s*%", pct, s)
    s = re.sub(r"\d+(?:\.\d+)?", lambda m: year_or_num(m) if "." not in m.group(0)
               else num2words(float(m.group(0))), s)
    return s


def normalise_word(w: str) -> str:
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z']", "", w)
    return w.strip("'")


def tokenize_reference(passage: str) -> list[RefWord]:
    """Split the passage into words, remembering where punctuation allows a pause."""
    out: list[RefWord] = []
    for raw in _expand_numbers(passage).split():
        # hyphenated compounds ("sea-level") are read as separate words
        parts = [p for p in re.split(r"[-–—/]", raw) if p]
        for i, p in enumerate(parts):
            norm = normalise_word(p)
            if not norm:
                if out:
                    out[-1].break_after = True
                continue
            last = i == len(parts) - 1
            out.append(RefWord(norm, p, bool(last and _PUNCT_BREAK.search(raw))))
    return out


def tokenize_hypothesis(text: str) -> list[str]:
    words = []
    for raw in _expand_numbers(text).split():
        for p in re.split(r"[-–—/]", raw):
            n = normalise_word(p)
            if n:
                words.append(n)
    return words
