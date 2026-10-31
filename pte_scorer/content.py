"""Content: align what was said against the passage (omissions/insertions/substitutions)."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .text import FILLERS


@dataclass
class AlignOp:
    op: str                 # "match" | "sub" | "del" (omitted) | "ins" (inserted)
    ref_idx: int | None
    hyp_idx: int | None


def align_words(ref: list[str], hyp: list[str]) -> list[AlignOp]:
    """Levenshtein alignment at word level with backtrace.

    Costs: insertion/deletion 0.75; substitution 1 - 0.6 * char-similarity, so a near-miss
    ("ecosis" for "ecosystems") is cheaper than pairing unrelated words. Making indels cheaper than a pair of unrelated substitutions keeps a
    skipped or repeated word from shifting every following word into a substitution chain.
    """
    n, m = len(ref), len(hyp)
    INDEL = 0.75

    def sub_cost(a: str, b: str) -> float:
        if a == b:
            return 0.0
        return 1.0 - 0.6 * SequenceMatcher(None, a, b).ratio()

    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * INDEL
    for j in range(1, m + 1):
        d[0][j] = j * INDEL
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]),
                          d[i - 1][j] + INDEL,
                          d[i][j - 1] + INDEL)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(d[i][j] - (d[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]))) < 1e-9:
            ops.append(AlignOp("match" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(d[i][j] - (d[i - 1][j] + INDEL)) < 1e-9:
            ops.append(AlignOp("del", i - 1, None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def content_features(ref: list[str], hyp: list[str]) -> tuple[dict, list[AlignOp]]:
    ops = align_words(ref, hyp)
    n_ref = max(len(ref), 1)
    n_match = sum(o.op == "match" for o in ops)
    n_sub = sum(o.op == "sub" for o in ops)
    n_del = sum(o.op == "del" for o in ops)
    ins = [hyp[o.hyp_idx] for o in ops if o.op == "ins"]
    n_filler = sum(w in FILLERS for w in ins)
    # an inserted word identical to a neighbouring hypothesis word = repetition
    n_rep = 0
    for o in ops:
        if o.op == "ins" and hyp[o.hyp_idx] not in FILLERS:
            j = o.hyp_idx
            if (j > 0 and hyp[j - 1] == hyp[j]) or (j + 1 < len(hyp) and hyp[j + 1] == hyp[j]):
                n_rep += 1
    n_ins = len(ins) - n_filler  # fillers are a fluency issue, not content

    # longest run of consecutive omitted words (skipping a whole phrase/line)
    longest_del, cur = 0, 0
    for o in ops:
        cur = cur + 1 if o.op == "del" else (cur if o.op == "ins" else 0)
        longest_del = max(longest_del, cur)

    # how far into the passage the speaker got (stopped early -> trailing omissions)
    last_read = max((o.ref_idx for o in ops if o.op in ("match", "sub")), default=-1)

    feats = {
        "n_ref_words": len(ref),
        "n_hyp_words": len(hyp),
        "pct_match": n_match / n_ref,
        "pct_sub": n_sub / n_ref,
        "pct_omit": n_del / n_ref,
        "pct_insert": n_ins / n_ref,
        "n_errors": n_sub + n_del + n_ins,
        "wer": (n_sub + n_del + n_ins) / n_ref,
        "longest_omission_run": longest_del,
        "pct_passage_covered": (last_read + 1) / n_ref,
        "n_fillers": n_filler,
        "n_repetitions": n_rep,
    }
    return feats, ops
