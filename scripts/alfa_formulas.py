"""Check the formulas recovered for the source's Read Aloud / Repeat Sentence trait scores.

Inputs are the source's own word colours (word_labels.jsonl): G = CORRECT, A = AVERAGE, R = WRONG,
n = G + A + R (words of the passage / sentence). Fluency also needs the recording length in seconds,
taken from a features parquet (column aq_duration_s).

    python scripts/alfa_formulas.py data/alfa
    python scripts/alfa_formulas.py data/alfa_rs --features cache/alfa_rs_r1_features.parquet --task repeat_sentence
"""
import argparse
import json

import numpy as np
import pandas as pd


def rnd(x):
    """Round half up, as the source does."""
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def pronunciation(G, A, R, n):
    """Both tasks: green +1, yellow +0.5, red -0.5 per word, scaled to 90, floored at 0."""
    return np.clip(rnd(90 * (G + 0.5 * A - 0.5 * R) / n), 0, 90)


def content_read_aloud(G, n):
    """Read Aloud: share of green words, scaled to 90."""
    return rnd(90 * G / n)


def fluency_read_aloud(G, n, seconds, k=35.8):
    """Read Aloud (approximate): content, capped by green words per second of recording.
    Full fluency needs about 2.5 passage words per second."""
    return np.clip(rnd(np.minimum(90 * G / n, k * G / np.maximum(seconds, 0.5))), 0, 90)


def content_table(df):
    """Repeat Sentence content is a fixed lookup on (G, n), not a straight line. Build it from data."""
    return df.groupby(["n", "G"]).content.agg(lambda s: s.mode().iloc[0])


def load(pool):
    lab = pd.read_csv(f"{pool}/labels.csv")
    rows = {}
    with open(f"{pool}/word_labels.jsonl") as fh:
        for line in fh:
            w = json.loads(line)
            if not w["words"]:
                continue
            labels = [lbl for _, lbl in w["words"]]
            rows[w["response_id"]] = dict(G=labels.count("CORRECT"), A=labels.count("AVERAGE"),
                                          R=labels.count("WRONG"), n=sum(lbl != "PAUSE" for lbl in labels))
    d = lab.merge(pd.DataFrame.from_dict(rows, orient="index"), left_on="response_id", right_index=True)
    d = d.dropna(subset=["content", "fluency", "pronunciation"])
    return d[(d.is_inflated != 1) & (d.n > 0)]


def report(name, pred, true):
    err = np.abs(np.asarray(pred) - np.asarray(true))
    print(f"  {name:14s} exact {np.mean(err == 0):6.1%}   within 1 {np.mean(err <= 1):6.1%}   "
          f"within 2 {np.mean(err <= 2):6.1%}   MAE {err.mean():.2f}   PCC {np.corrcoef(pred, true)[0, 1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--task", default="read_aloud", choices=["read_aloud", "repeat_sentence"])
    ap.add_argument("--features", help="features parquet with aq_duration_s, for fluency")
    a = ap.parse_args()

    d = load(a.pool)
    print(f"{a.task}: {len(d)} answers")
    report("pronunciation", pronunciation(d.G, d.A, d.R, d.n), d.pronunciation)
    if a.task == "read_aloud":
        report("content", content_read_aloud(d.G, d.n), d.content)
    else:
        tab = content_table(d)
        report("content", tab.reindex(pd.MultiIndex.from_arrays([d.n, d.G])).values, d.content)

    if a.features:
        f = pd.read_parquet(a.features, columns=["response_id", "aq_duration_s"])
        m = d.merge(f, on="response_id").dropna(subset=["aq_duration_s"])
        m = m[(m.content + m.fluency + m.pronunciation) / 3 >= 10]
        if a.task == "read_aloud":
            report("fluency", fluency_read_aloud(m.G, m.n, m.aq_duration_s), m.fluency)
        else:
            print("  fluency        no closed form yet: depends on G/n and answer length / prompt audio length "
                  "(data/alfa_rs/prompt_durations.csv); see docs/data_pipeline.md")


if __name__ == "__main__":
    main()
