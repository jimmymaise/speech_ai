"""Train / evaluate / load the feature -> trait score models (Level A: LightGBM)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

from . import TRAITS

# Model inputs are an allow-list: features computed by the pipeline from the audio
# (listed in <parquet>.features.json by `extract`) plus these context columns.
# Anything else in labels.csv (other scores from the label source, counts, ids) is never an input.
CONTEXT_FEATURES = {"pte_type": {"academic": 1, "core": 2}}

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def feature_columns(df: pd.DataFrame, audio_features: list[str]) -> list[str]:
    cols = [c for c in audio_features if c in df.columns]
    return cols + [c for c in CONTEXT_FEATURES if c in df.columns and df[c].nunique() > 1]


def audio_feature_list(features_parquet: str | Path) -> list[str]:
    side = Path(str(features_parquet) + ".features.json")
    if not side.exists():
        raise FileNotFoundError(f"{side} missing: re-run `extract` (it lists which columns are audio features)")
    return json.loads(side.read_text())


def metrics(y: np.ndarray, p: np.ndarray, hi: float = 5.0) -> dict:
    """Errors are also reported on a 0-5 equivalent so 0-5 and 0-90 labels are comparable."""
    k = 5.0 / hi
    yi, pi = np.round(y * k).astype(int), np.clip(np.round(p * k), 0, 5).astype(int)
    y, p = y * k, p * k
    return {
        "pcc": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else float("nan"),
        "mae": float(mean_absolute_error(y, p)),
        "qwk": float(cohen_kappa_score(yi, pi, weights="quadratic")) if len(set(yi)) > 1 else float("nan"),
        "within_0_5": float(np.mean(np.abs(y - p) <= 0.5)),
        "within_1": float(np.mean(np.abs(y - p) <= 1.0)),
        "n": int(len(y)),
    }


def _new_model():
    import lightgbm as lgb
    return lgb.LGBMRegressor(**LGB_PARAMS)


def cross_validate(df: pd.DataFrame, trait: str, cols: list[str], group_by: str = "speaker_id",
                   folds: int = 5) -> tuple[dict, np.ndarray]:
    d = df[df[trait].notna()]
    X, y, g = d[cols], d[trait].to_numpy(float), d[group_by].to_numpy()
    hi = label_max(y)
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        m = _new_model().fit(X.iloc[tr], y[tr])
        oof[te] = np.clip(m.predict(X.iloc[te]), 0, hi)
    res = metrics(y, oof, hi)
    # per PTE type, so a weak Core model is not hidden behind a good overall number
    if "pte_type" in d.columns and d["pte_type"].nunique() > 1:
        names = {v: k for k, v in CONTEXT_FEATURES["pte_type"].items()}
        for v in sorted(d["pte_type"].dropna().unique()):
            mask = (d["pte_type"] == v).to_numpy()
            res[f"by_type_{names.get(int(v), v)}"] = metrics(y[mask], oof[mask], hi)
    return res, oof


def drop_label_failures(df: pd.DataFrame, traits=TRAITS) -> pd.DataFrame:
    """Alfa sometimes scores a complete, correct reading as 0/0/0 (scoring failure, ~30% of its zeros).
    Drop answers with all traits < 10 (0-90) while our ASR heard >= 50% of the passage read correctly."""
    if "pct_match" not in df.columns or not all(t in df.columns for t in traits):
        return df
    lvl = df[list(traits)].mean(axis=1)
    hi = 90.0 if lvl.max() > 10 else 5.0
    bad = (lvl < 10 * hi / 90) & (df["pct_match"] >= 0.5)
    if bad.any():
        print(f"dropped {int(bad.sum())} answers scored ~0 by the label source although read correctly")
    return df[~bad]


def label_max(y: np.ndarray) -> float:
    """Labels are either PTE trait scores (0-5) or 0-90 scaled scores."""
    return 5.0 if np.nanmax(y) <= 5 else 90.0


def train(features_parquet: str | Path, out_path: str | Path, traits=TRAITS,
          group_by: str = "speaker_id", folds: int = 5) -> dict:
    df = drop_label_failures(pd.read_parquet(features_parquet))
    cols = feature_columns(df, audio_feature_list(features_parquet))
    bundle = {"features": cols, "models": {}, "cv": {}, "group_by": group_by, "scale": {}}
    for t in traits:
        if t not in df.columns or df[t].notna().sum() < 50:
            print(f"[{t}] skipped: not enough labels")
            continue
        hi = label_max(df[t].to_numpy(float))
        if df[t].std() < 0.06 * hi:
            print(f"[{t}] skipped: labels almost constant (std={df[t].std():.2f}); rule-based scoring is used")
            continue
        cv, _ = cross_validate(df, t, cols, group_by, folds)
        bundle["cv"][t] = cv
        d = df[df[t].notna()]
        bundle["models"][t] = _new_model().fit(d[cols], d[t].to_numpy(float))
        bundle["scale"][t] = hi
        flat = {k: v for k, v in cv.items() if not isinstance(v, dict)}
        print(f"[{t}] (0-{hi:g}) {folds}-fold CV grouped by {group_by}, errors on 0-5 scale: "
              + ", ".join(f"{k}={v:.3f}" for k, v in flat.items()))
        for k, sub in cv.items():
            if isinstance(sub, dict):
                print(f"      {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in sub.items()))
    joblib.dump(bundle, out_path)
    Path(str(out_path) + ".metrics.json").write_text(json.dumps(bundle["cv"], indent=2))
    return bundle


def top_features(bundle: dict, trait: str, k: int = 15) -> list[tuple[str, float]]:
    m = bundle["models"][trait]
    imp = m.booster_.feature_importance("gain")
    order = np.argsort(imp)[::-1][:k]
    return [(bundle["features"][i], float(imp[i] / imp.sum())) for i in order]


def load(path: str | Path) -> dict | None:
    p = Path(path)
    return joblib.load(p) if p.exists() else None
