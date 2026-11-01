"""Check the formulas recovered for APEUni's Read Aloud scores (see docs/apeuni_formulas.md).

Input: a decoded APEUni `answers/list` dump, {"answers": [...]} (one or many pages merged).

    python scripts/apeuni_formulas.py /path/to/answers-1519/answers.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

WORD_K = 83  # word score = floor(83 * ai_score)
SOUND_GOOD, SOUND_AVG = 80, 70
WORD_GOOD, WORD_AVG = 71, 63


def sound_score(origin_score):
    return np.floor(100 * np.asarray(origin_score, dtype=float))


def word_score(ai_score):
    return np.floor(WORD_K * np.asarray(ai_score, dtype=float))


def read_aloud_score(content, pronunciation, fluency):
    """Content (0-5) is a gate: 5 -> (P+F)/2, 4 -> x0.9, 0 -> 10. Levels 1-3 not seen."""
    base = (np.asarray(pronunciation) + np.asarray(fluency)) / 2
    factor = np.select([content >= 5, content == 4], [1.0, 0.9], default=np.nan)
    return np.where(content == 0, 10, np.floor(base * factor + 0.5))


def flatten(path):
    d = json.load(open(path))
    rows, words = [], []
    for x in d["answers"]:
        a = x["attributes"]
        c = a["scores"]["components"]
        det = a["score_details"]["details"]
        cw = [t for t in det if t["type"] == "content_word"]
        for t in cw:
            words.append((t["ai_score"], t["score"], [(s["origin_score"], s["score"]) for s in t["syllables"]]))
        dur = np.array([(t["end_pos"] - t["beg_pos"]) / 100 for t in cw])
        ph = np.array([s["score"] for t in cw for s in t["syllables"]])
        rows.append(dict(
            total=a["total_score"], C=c["content"]["score"], P=c["pronunciation"]["score"], F=c["fluency"]["score"],
            engine=a["algo_version_tag"]["tag_name"], n_words=len(cw),
            n_omit=sum(t["type"] == "omission" for t in det),
            seconds=dur.sum() if len(cw) else np.nan, sound_mean=ph.mean() if len(ph) else np.nan,
        ))
    return pd.DataFrame(rows), words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers_json")
    a = ap.parse_args()
    df, words = flatten(a.answers_json)
    print(f"{len(df)} answers, engines {df.engine.value_counts().to_dict()}")

    ai = np.array([w[0] for w in words]); ws = np.array([w[1] for w in words])
    so = np.array([s[0] for w in words for s in w[2]]); ss = np.array([s[1] for w in words for s in w[2]])
    print(f"word score = floor(83*ai_score): {np.mean(word_score(ai) == ws):.1%} of {len(ws)} words")
    print(f"sound score = floor(100*origin): {np.mean(sound_score(so) == ss):.1%} of {len(ss)} sounds")

    pred = read_aloud_score(df.C.values, df.P.values, df.F.values)
    for c, g in df.assign(ok=pred == df.total).groupby("C"):
        print(f"Read Aloud score, content {c}: {g.ok.mean():.1%} of {len(g)}")
    print("content by omitted words:", df.groupby("C").n_omit.agg(["min", "max"]).to_dict("index"))

    full = df[(df.C == 5) & (df.n_words == df.n_words.max())].dropna()
    for target, cols in [("F", ["seconds"]), ("P", ["sound_mean", "F"])]:
        p = cross_val_predict(LinearRegression(), full[cols], full[target], cv=5)
        print(f"{target} ~ {cols}: PCC {np.corrcoef(p, full[target])[0, 1]:.3f}, MAE {np.abs(p - full[target]).mean():.2f}")


if __name__ == "__main__":
    main()
