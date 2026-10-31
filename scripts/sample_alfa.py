"""Pick a representative, de-duplicated subset of crawled responses to download and train on.

  python scripts/sample_alfa.py data/alfa --per-band 8 --max-per-student 15

Rules
  1. drop: missing scores, is_inflated == 1
  2. one attempt per (student, passage): a random attempt, not the best one (best would bias scores upward)
  3. level bands on the mean of content/fluency/pronunciation (0-90); per passage take up to --per-band
     responses from each band, so weak and strong levels are both covered
  4. a student contributes at most --max-per-student responses overall
  4b. within each (passage, band) about --core-share of the slots go to PTE Core responses (if available);
      unused Core slots are back-filled with Academic, so no data is wasted
  5. split by student hash into train / val / test (80/10/10); ~10% of passages are held out as
     test_passage and only answered by val/test students -> no passage or student shared with train
Writes <root>/selected.csv (subset of labels.csv + band + split) and prints the coverage tables.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# PTE overall-score style bands (roughly CEFR: <30 pre-A2, 30-42 A2, 43-58 B1, 59-75 B2, 76-84 C1, 85+ C2)
BANDS = [(0, 10, "00-09 silent/none"), (10, 30, "10-29"), (30, 43, "30-42 A2"), (43, 51, "43-50 B1-"),
         (51, 59, "51-58 B1+"), (59, 66, "59-65 B2-"), (66, 76, "66-75 B2+"), (76, 85, "76-84 C1"),
         (85, 91, "85-90 C2")]


def band_of(x: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= x < hi:
            return name
    return BANDS[-1][2]


def bucket(key: str, salt: str) -> float:
    """Deterministic pseudo-random number in [0,1) from a key."""
    return int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def select(df: pd.DataFrame, per_band: int, max_per_student: int, seed: int,
           core_share: float = 1 / 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["content", "fluency", "pronunciation"])
    d = d[d.get("is_inflated", 0).fillna(0) != 1].copy()
    d["level"] = d[["content", "fluency", "pronunciation"]].mean(axis=1)
    d["band"] = d["level"].map(band_of)
    d["_r"] = rng.random(len(d))

    # (2) one random attempt per student per passage
    d = d.sort_values("_r").drop_duplicates(["speaker_id", "passage_id"])

    # (3)+(4) per passage and band, fill with students who still have quota (rarest bands first,
    # so a student's quota is not used up on the common bands)
    band_size = d.band.value_counts()
    d["_band_rank"] = d.band.map(band_size)
    d = d.sort_values(["_band_rank", "_r"])
    used: dict[str, int] = {}
    taken: dict[tuple, int] = {}
    core_quota = int(round(per_band * core_share))
    quotas = {2: core_quota, 1: per_band - core_quota}   # pte_type: 1 academic, 2 core
    keep, kept = [], set()

    def take(rows, cap_key, cap):
        for idx, pid, band, sid, typ in rows:
            if idx in kept or used.get(sid, 0) >= max_per_student:
                continue
            k = cap_key(pid, band, typ)
            if taken.get(k, 0) >= cap(typ):
                continue
            taken[k] = taken.get(k, 0) + 1
            used[sid] = used.get(sid, 0) + 1
            keep.append(idx)
            kept.add(idx)

    rows = list(zip(d.index, d.passage_id, d.band, d.speaker_id, d.get("pte_type", pd.Series(1, index=d.index))))
    # pass 1: typed quotas
    take(rows, lambda p, b, t: (p, b, t), lambda t: quotas.get(t, per_band))
    # pass 2: fill the remaining band slots with any type
    filled = d.loc[keep].groupby(["passage_id", "band"]).size().to_dict()
    for k, v in filled.items():
        taken[("any",) + k] = v
    take(rows, lambda p, b, t: ("any", p, b), lambda t: per_band)
    return d.loc[keep].drop(columns=["_r", "_band_rank"])


def student_bucket(sid: str, seed: int) -> float:
    return bucket(sid, f"{seed}s")


def held_out_passages(passages, frac: float, seed: int) -> set:
    return {p for p in passages if bucket(p, f"{seed}p") < frac} if len(passages) > 10 else set()


def split_of(pid: str, sid: str, held: set, seed: int) -> str:
    """Students are hashed to train (80%) / val (10%) / test (10%). Held-out passages only take
    val/test students, so `test_passage` shares neither passages nor students with train."""
    if pid in held:
        return "test_passage"
    u = student_bucket(sid, seed)
    return "train" if u < 0.8 else ("val" if u < 0.9 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--per-band", type=int, default=8, help="max responses per passage per level band")
    ap.add_argument("--max-per-student", type=int, default=15)
    ap.add_argument("--holdout-passages", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--core-share", type=float, default=1 / 3, help="target share of PTE Core per band")
    a = ap.parse_args()

    root = Path(a.root)
    df = pd.read_csv(root / "labels.csv")
    held = held_out_passages(df.passage_id.unique(), a.holdout_passages, a.seed)
    # train students never answer a held-out passage in the selection
    cand = df[~(df.passage_id.isin(held) & (df.speaker_id.map(lambda x: student_bucket(x, a.seed)) < 0.8))]
    sel = select(cand, a.per_band, a.max_per_student, a.seed, a.core_share)
    sel["split"] = [split_of(p, sid, held, a.seed) for p, sid in zip(sel.passage_id, sel.speaker_id)]
    sel.to_csv(root / "selected.csv", index=False)

    pool = df.dropna(subset=["content", "fluency", "pronunciation"]).copy()
    pool["band"] = pool[["content", "fluency", "pronunciation"]].mean(axis=1).map(band_of)
    print(f"pool: {len(df)} responses, {df.speaker_id.nunique()} students, {df.passage_id.nunique()} passages")
    print(f"selected: {len(sel)} responses, {sel.speaker_id.nunique()} students, "
          f"max {sel.speaker_id.value_counts().max()} per student\n")
    tab = pd.DataFrame({"pool": pool.band.value_counts(), "selected": sel.band.value_counts(),
                        "selected_core": sel[sel.pte_type == 2].band.value_counts()}).fillna(0).astype(int)
    print(tab.sort_index().to_string(), "\n")
    print("by PTE type:", sel.pte_type.map({1: "academic", 2: "core"}).value_counts().to_dict())
    print("by split:", sel.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
