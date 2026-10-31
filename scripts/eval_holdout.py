"""Train on the `train` split, report on `test` (new students) and `test_passage` (new students AND
new passages) against the "like Alfa" targets. Scores are 0-90 points.

  python scripts/eval_holdout.py cache/alfa_r1_features.parquet [--curve] [--out report.json]

Targets: PCC >= .90 per trait, MAE <= 6 points, >= 85% within +-10 points, >= 75% same CEFR level.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pte_scorer import TRAITS  # noqa: E402
from pte_scorer import model as M  # noqa: E402
from pte_scorer.levels import cefr, to_pte_scale  # noqa: E402

TYPES = {1: "academic", 2: "core"}


def point_metrics(y, p) -> dict:
    return {"pcc": float(np.corrcoef(y, p)[0, 1]), "mae": float(np.mean(np.abs(y - p))),
            "within_10": float(np.mean(np.abs(y - p) <= 10)), "n": int(len(y))}


def report(d: pd.DataFrame, pred: dict) -> dict:
    out = {t: point_metrics(d[t].to_numpy(float), pred[t]) for t in TRAITS}
    y_lvl = d[list(TRAITS)].mean(axis=1).map(lambda v: cefr(to_pte_scale(v, 90)))
    p_lvl = pd.Series(np.mean([pred[t] for t in TRAITS], axis=0), index=d.index).map(
        lambda v: cefr(to_pte_scale(v, 90)))
    out["same_cefr"] = float((y_lvl == p_lvl).mean())
    out["cefr_off_by_more_than_1"] = float((abs(y_lvl.map(ORDER) - p_lvl.map(ORDER)) > 1).mean())
    # on 43+ (where most real users are) the easy silent/garbage answers do not inflate PCC
    m = (d[list(TRAITS)].mean(axis=1) >= 43).to_numpy()
    out["level_43plus"] = {t: point_metrics(d[t].to_numpy(float)[m], pred[t][m]) for t in TRAITS}
    return out


ORDER = {k: i for i, k in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"])}


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> dict:
    pred = {}
    for t in TRAITS:
        m = M._new_model().fit(tr[cols], tr[t].to_numpy(float))
        pred[t] = np.clip(m.predict(te[cols]), 0, 90)
    return pred


def show(name: str, r: dict):
    print(f"\n== {name}  (n={r['content']['n']})")
    print(f"   {'trait':14s} {'PCC':>6s} {'MAE':>6s} {'±10':>6s} | 43+: {'PCC':>6s} {'MAE':>6s} {'±10':>6s}")
    for t in TRAITS:
        a, b = r[t], r["level_43plus"][t]
        print(f"   {t:14s} {a['pcc']:6.3f} {a['mae']:6.2f} {a['within_10']:6.1%} |      "
              f"{b['pcc']:6.3f} {b['mae']:6.2f} {b['within_10']:6.1%}")
    print(f"   same CEFR level: {r['same_cefr']:.1%}   off by >1 level: {r['cefr_off_by_more_than_1']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--curve", action="store_true", help="also train on 25/50%% of train students")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    df = M.drop_label_failures(df.dropna(subset=list(TRAITS)))
    if "error" in df.columns:
        df = df[df["error"].isna()]
    cols = M.feature_columns(df, M.audio_feature_list(a.features))
    tr = df[df.split == "train"]
    full = {}
    for name in ("test", "test_passage"):
        te = df[df.split == name]
        pred = fit_predict(tr, te, cols)
        r = report(te, pred)
        show(f"{name}: all", r)
        full[name] = {"all": r}
        for v, tn in TYPES.items():
            mask = (te.pte_type == v).to_numpy()
            if mask.sum() > 30:
                rt = report(te[mask], {t: p[mask] for t, p in pred.items()})
                show(f"{name}: {tn}", rt)
                full[name][tn] = rt

    if a.curve:
        te = df[df.split.isin(["test", "test_passage"])]
        students = tr.speaker_id.unique()
        rng = np.random.default_rng(0)
        print("\n== learning curve (test + test_passage, PCC content/fluency/pron, MAE pron)")
        for frac in (0.25, 0.5, 1.0):
            keep = set(rng.choice(students, int(len(students) * frac), replace=False))
            sub = tr[tr.speaker_id.isin(keep)]
            r = report(te, fit_predict(sub, te, cols))
            print(f"   {frac:4.0%} of train ({len(sub):5d} responses): "
                  + " ".join(f"{r[t]['pcc']:.3f}" for t in TRAITS) + f"  mae_pron {r['pronunciation']['mae']:.2f}")
            full[f"curve_{frac}"] = r
    if a.out:
        Path(a.out).write_text(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()
