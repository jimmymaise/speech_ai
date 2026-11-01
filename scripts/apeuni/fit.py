"""Check the recovered score rules and fit the approximate traits for one flattened task.

    python scripts/apeuni/flatten.py <task> && python scripts/apeuni/fit.py <task>

All model fits hold out whole questions (GroupKFold by question).
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from common import STOP, out_path

# Task score rules recovered from the data (docs/apeuni_formulas.md).
RULES = {
    "describe_image": lambda d: (d.P + d.F) / 2,
    "respond_situation": lambda d: (d.P + d.F) / 2 * d.C / 6,
    "retell_lecture": lambda d: 10 + 8 * (d.P + d.F),
    "repeat_sentence": lambda d: np.where(np.isclose(d.C, 3), 47.06 + 0.2422 * (d.P + d.F), np.nan),
}


def tok(s):
    return [w for w in re.findall(r"[a-z']+", (s or "").lower()) if w not in STOP and len(w) > 2]


def add_features(d):
    kw = []
    for _, r in d.iterrows():
        rt, at = set(tok(r.get("ref"))), tok(r.get("rec"))
        kw.append(dict(kw_hit=len(set(at) & rt), kw_frac=len(set(at) & rt) / max(1, len(rt)), uniq=len(set(at)),
                       n_tok=len(at)))
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(kw)], axis=1)
    if "seconds" in d and d.seconds.notna().any():
        d["wps"] = d.n_words / d.seconds
        d["lps"] = d.letters / d.seconds
    return d


def main():
    task = sys.argv[1]
    d = add_features(pd.read_parquet(out_path(f"{task}.parquet")))
    cv = GroupKFold(d.q.nunique())

    def ev(target, cols, model, name):
        cols = list(dict.fromkeys(c for c in cols if c in d and d[c].notna().all()))
        if not cols:
            return
        p = cross_val_predict(model, d[cols], d[target], cv=cv, groups=d.q)
        print(f"  {target:5s} {name:26s} PCC {np.corrcoef(p, d[target])[0, 1]:.3f}  MAE {np.abs(p - d[target]).mean():.2f}")

    gbm = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    print(f"{task}: {len(d)} answers, {d.q.nunique()} questions")
    if task in RULES:
        p = pd.Series(RULES[task](d)); ok = p.notna()
        e = np.abs(np.floor(p[ok] + 0.5) - d.total[ok])
        print(f"  task score rule: exact {np.mean(e == 0):.1%}, within 1 {np.mean(e <= 1):.1%} (n={ok.sum()})")
    ev("total", ["C", "P", "F"], gbm(), "model on C, P, F")
    ev("P", ["sound_mean", "F"], LinearRegression(), "sound mean + F")
    ev("P", ["word_mean", "F"], LinearRegression(), "word mean + F")
    ev("F", ["wps", "lps", "n_words", "seconds", "n_long1", "d_std", "tc_fluency", "tcf_low"], gbm(), "timing / length")
    ev("F", ["wps", "lps", "n_words", "seconds", "n_long1", "d_std", "tc_fluency", "sound_mean", "word_mean", "C"],
       gbm(), "+ pronunciation + content")
    ev("C", ["kw_hit", "kw_frac"], gbm(), "keyword overlap")
    ev("C", ["kw_hit", "kw_frac", "uniq", "n_tok", "n_words"], gbm(), "+ length")
    if task == "respond_situation":
        p = np.minimum(6, np.round(0.12 * d.n_words, 1))
        print(f"  content = min(6, 0.12 x words): exact {np.mean(np.isclose(p, d.C)):.1%}")


if __name__ == "__main__":
    main()
