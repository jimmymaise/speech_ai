"""Dataset loading, validation and resumable batch feature extraction.

Dataset layout (any directory):
  labels.csv    response_id, passage_id, speaker_id, audio_path, content, fluency, pronunciation [, flags, rater_id, ...]
  passages.csv  passage_id, reference_text
audio_path is relative to the dataset directory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import TRAITS
from .audio import audio_quality, load_audio

REQUIRED = ["response_id", "passage_id", "speaker_id", "audio_path"]


def load_dataset(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    labels = pd.read_csv(root / "labels.csv", dtype={"response_id": str, "passage_id": str, "speaker_id": str})
    passages = pd.read_csv(root / "passages.csv", dtype={"passage_id": str})
    df = labels.merge(passages, on="passage_id", how="left")
    df["audio_abspath"] = df["audio_path"].map(lambda p: str(root / p))
    return df


def validate(root: str | Path, check_audio: bool = True) -> pd.DataFrame:
    """Return one row per problem found. Empty frame = dataset is clean."""
    root = Path(root)
    problems = []

    def bad(rid, issue):
        problems.append({"response_id": rid, "issue": issue})

    labels = pd.read_csv(root / "labels.csv", dtype=str)
    passages = pd.read_csv(root / "passages.csv", dtype=str)
    for c in REQUIRED:
        if c not in labels.columns:
            bad("*", f"labels.csv missing column {c}")
    if problems:
        return pd.DataFrame(problems)
    for t in TRAITS:
        if t not in labels.columns:
            bad("*", f"labels.csv missing score column {t} (allowed only for unlabeled data)")
    for rid in labels.response_id[labels.response_id.duplicated()]:
        bad(rid, "duplicate response_id")
    known = set(passages.passage_id)
    for pid, n in passages.passage_id.value_counts().items():
        if n > 1:
            bad("*", f"passage {pid} defined {n} times")
    for _, r in passages.iterrows():
        if not isinstance(r.reference_text, str) or len(r.reference_text.split()) < 5:
            bad("*", f"passage {r.passage_id} has empty/very short text")

    flags = labels["flags"].fillna("") if "flags" in labels.columns else pd.Series("", index=labels.index)
    # each score column is either PTE trait scale (0-5) or 0-90 scaled scores (max > 10)
    scale = {t: (90 if pd.to_numeric(labels[t], errors="coerce").max() > 10 else 5)
             for t in TRAITS if t in labels.columns}
    for i, r in labels.iterrows():
        rid = r.response_id
        if r.passage_id not in known:
            bad(rid, f"unknown passage_id {r.passage_id}")
        if not isinstance(r.speaker_id, str) or not r.speaker_id.strip():
            bad(rid, "missing speaker_id")
        for t in TRAITS:
            if t in labels.columns:
                v = pd.to_numeric(r[t], errors="coerce")
                if pd.isna(v) and not flags[i]:
                    bad(rid, f"missing {t} score and no flag")
                elif not pd.isna(v) and not 0 <= v <= scale[t]:
                    bad(rid, f"{t}={v} outside 0-{scale[t]}")
        p = root / str(r.audio_path)
        if not p.exists():
            bad(rid, f"audio not found: {r.audio_path}")
        elif check_audio:
            try:
                q = audio_quality(load_audio(str(p)))
            except Exception as e:  # noqa: BLE001
                bad(rid, f"audio unreadable: {e}")
                continue
            if q["duration_s"] < 1:
                bad(rid, f"audio very short ({q['duration_s']:.1f}s)")
            if q["duration_s"] > 45:
                bad(rid, f"audio longer than 40s ({q['duration_s']:.0f}s)")
            if q["clipping_ratio"] > 0.001:
                bad(rid, f"clipping ({q['clipping_ratio']:.2%} samples)")
            if q["rms_db"] < -50:
                bad(rid, f"very quiet ({q['rms_db']:.0f} dBFS)")
    return pd.DataFrame(problems, columns=["response_id", "issue"])


def dataset_report(root: str | Path) -> str:
    df = load_dataset(root)
    lines = [f"responses: {len(df)}", f"passages: {df.passage_id.nunique()}",
             f"speakers: {df.speaker_id.nunique()}",
             f"max responses per speaker: {df.speaker_id.value_counts().max()}",
             f"responses per passage: median {df.passage_id.value_counts().median():.0f}"]
    for t in TRAITS:
        if t in df.columns:
            vc = pd.to_numeric(df[t], errors="coerce").round().value_counts().sort_index()
            lines.append(f"{t} distribution (rounded): " + ", ".join(f"{int(k)}:{v}" for k, v in vc.items()))
    return "\n".join(lines)


def extract_features(root: str | Path, cache_dir: str | Path, out_parquet: str | Path,
                     limit: int | None = None, analyzer=None) -> pd.DataFrame:
    """Run the analyzer on every response. One JSON per response in cache_dir, so a restart resumes."""
    from .pipeline import Analyzer

    df = load_dataset(root)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in df.itertuples() if not (cache / f"{r.response_id}.json").exists()]
    print(f"{len(df) - len(todo)} cached, {len(todo)} to process", flush=True)
    if todo:
        analyzer = analyzer or Analyzer()
        t0 = time.time()
        for k, r in enumerate(todo, 1):
            try:
                res = analyzer.analyze(r.audio_abspath, r.reference_text)
            except Exception as e:  # noqa: BLE001  keep going; record the failure
                res = {"features": {}, "details": {"error": repr(e)}}
            (cache / f"{r.response_id}.json").write_text(json.dumps(res))
            if k % 25 == 0 or k == len(todo):
                rate = (time.time() - t0) / k
                print(f"{k}/{len(todo)}  {rate:.2f}s/clip  eta {rate * (len(todo) - k) / 60:.0f} min", flush=True)

    rows = []
    for r in df.itertuples():
        res = json.loads((cache / f"{r.response_id}.json").read_text())
        rows.append({"response_id": r.response_id, **res["features"]})
    feats = pd.DataFrame(rows)
    keep = [c for c in df.columns if c not in ("audio_abspath",)]
    out = df[keep].merge(feats, on="response_id", how="left")
    out.to_parquet(out_parquet, index=False)
    Path(str(out_parquet) + ".features.json").write_text(
        json.dumps([c for c in feats.columns if c != "response_id"]))
    print(f"wrote {out_parquet}: {len(out)} rows, {feats.shape[1] - 1} features")
    return out
