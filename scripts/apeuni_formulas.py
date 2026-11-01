"""Check the formulas recovered for APEUni's Read Aloud scores (see docs/apeuni_formulas.md).

Input: a folder with one sub-folder per question (`<id>/answers.json` + `<id>/questions.json`),
or a single decoded `answers.json`.

    python scripts/apeuni_formulas.py /path/to/read_aloud
"""
import os
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

WORD_K = 83  # word score = floor(83 * ai_score)
SOUND_GOOD, SOUND_AVG = 80, 70
WORD_GOOD, WORD_AVG = 71, 63


def sound_score(origin_score):
    return np.floor(100 * np.asarray(origin_score, dtype=float))


def word_score(ai_score):
    return np.floor(WORD_K * np.asarray(ai_score, dtype=float))


def read_aloud_score(content, pronunciation, fluency):
    """Content (0-5) is a gate: 5 -> (P+F)/2, 4 -> x0.9, 2 -> x0.55, 0 -> 10. Levels 1 and 3 not seen."""
    base = (np.asarray(pronunciation) + np.asarray(fluency)) / 2
    factor = np.select([content >= 5, content == 4, content == 2], [1.0, 0.9, 0.55], default=np.nan)
    return np.where(content == 0, 10, np.floor(base * factor + 0.5))


def answer_files(path):
    if os.path.isfile(path):
        return [(None, path)]
    out = []
    for q in sorted(os.listdir(path)):
        f = os.path.join(path, q, "answers.json")
        if q.isdigit() and os.path.exists(f):
            out.append((int(q), f))
    return out


def passage_letters(answers_path):
    qf = os.path.join(os.path.dirname(answers_path), "questions.json")
    if not os.path.exists(qf):
        return np.nan
    return sum(ch.isalpha() for ch in json.load(open(qf)).get("prediction_text", ""))


def flatten(path):
    rows, words = [], []
    for qid, f in answer_files(path):
        letters = passage_letters(f)
        rows_q, words_q = flatten_one(f, qid, letters)
        rows += rows_q
        words += words_q
    return pd.DataFrame(rows), words


def flatten_one(path, qid, letters):
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
            q=qid, letters=letters, total=a["total_score"], C=c["content"]["score"], P=c["pronunciation"]["score"], F=c["fluency"]["score"],
            engine=a["algo_version_tag"]["tag_name"], n_words=len(cw),
            n_omit=sum(t["type"] == "omission" for t in det),
            seconds=dur.sum() if len(cw) else np.nan, sound_mean=ph.mean() if len(ph) else np.nan,
        ))
    return rows, words


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

    full = df[(df.C == 5) & (df.n_omit == 0)].dropna(subset=["seconds", "sound_mean"]).copy()
    full["letters_per_s"] = full.letters / full.seconds
    groups = full.q.fillna(0)
    cv = GroupKFold(groups.nunique()) if groups.nunique() > 1 else 5
    for target, cols, model in [
        ("F", ["letters_per_s"], GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05)),
        ("P", ["sound_mean", "F"], LinearRegression()),
    ]:
        x = full.dropna(subset=cols)
        p = cross_val_predict(model, x[cols], x[target], cv=cv, groups=groups.loc[x.index])
        print(f"{target} ~ {cols} (questions held out): PCC {np.corrcoef(p, x[target])[0, 1]:.3f}, "
              f"MAE {np.abs(p - x[target]).mean():.2f}")


if __name__ == "__main__":
    main()
