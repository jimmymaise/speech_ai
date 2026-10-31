"""PTE 10-90 scale and CEFR level for one Read Aloud answer (an estimate, not a PTE score)."""
from __future__ import annotations

PTE_MIN, PTE_MAX = 10, 90

# Commonly cited PTE -> CEFR alignment. Pearson has revised this before: check the current
# official table before showing it to users and update here.
CEFR_BANDS = [  # (lowest PTE score in band, CEFR label)
    (85, "C2"),
    (76, "C1"),
    (59, "B2"),
    (43, "B1"),
    (30, "A2"),
    (10, "A1"),
]

NOTE = ("Estimated level for this single Read Aloud answer. A PTE score combines many tasks, "
        "so this is not your PTE score.")


def to_pte_scale(value: float, scale_max: float) -> float:
    """Map a trait value to 10-90. 0-90 model output is used as is (Alfa's 0 = no answer -> 10);
    a 0-5 trait score is mapped linearly (0 -> 10, 5 -> 90)."""
    if scale_max == 5.0:
        v = PTE_MIN + value / 5.0 * (PTE_MAX - PTE_MIN)
    else:
        v = value
    return float(min(PTE_MAX, max(PTE_MIN, v)))


def cefr(pte_score: float) -> str:
    for lo, label in CEFR_BANDS:
        if pte_score >= lo:
            return label
    return "A1"


def estimate(trait_10_90: dict[str, float]) -> dict:
    overall = sum(trait_10_90.values()) / len(trait_10_90)
    return {
        "traits_10_90": {t: int(round(v)) for t, v in trait_10_90.items()},
        "read_aloud_10_90": int(round(overall)),
        "cefr": cefr(overall),
        "note": NOTE,
    }
