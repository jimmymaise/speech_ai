"""Train the word classifier (CORRECT / AVERAGE / WRONG) on Alfa's word labels.

  python scripts/train_word_labels.py data/alfa_r1 cache/alfa_r1

Trains on train+val answers, reports agreement on test + test_passage, saves models/word_labels.joblib
(loaded automatically by pte_scorer.words). Answers Alfa scored ~0 although read correctly are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pte_scorer import words as W  # noqa: E402


def norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalnum() or c == "'")


def build(root: Path, cache: Path, alfa_words: Path) -> pd.DataFrame:
    lab = pd.read_csv(root / "labels.csv").set_index("response_id")
    ids = {r for r in lab.index if (cache / f"{r}.json").exists()}
    alfa = {}
    with open(alfa_words) as fh:
        for line in fh:
            r = json.loads(line)
            if r["response_id"] in ids:
                alfa[r["response_id"]] = [w for w in r["words"] if w[1] != "PAUSE"]
    rows = []
    for rid in ids:
        if rid not in alfa:
            continue
        a = json.loads((cache / f"{rid}.json").read_text())
        d, f = a["details"], a["features"]
        if not d.get("ref_words") or not any("phones" in w for w in d.get("words", [])):
            continue
        L = lab.loc[rid]
        level = np.mean([L.content, L.fluency, L.pronunciation])
        if level < 10 and f.get("pct_match", 0) >= 0.5:   # label-source scoring failure
            continue
        feats = W.word_feature_rows(d, f, int(L.pte_type))
        ours = [w["word"] for w in W.word_report(d, use_classifier=False)]
        aw = alfa[rid]
        sm = SequenceMatcher(None, [norm(w) for w in ours], [norm(w[0]) for w in aw], autojunk=False)
        for b in sm.get_matching_blocks():
            for j in range(b.size):
                rows.append({**feats[b.a + j], "rid": rid, "split": L.split, "level": level,
                             "y": W.LABELS.index(aw[b.b + j][1])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("cache")
    ap.add_argument("--alfa-words", default="data/alfa/word_labels.jsonl")
    a = ap.parse_args()
    import lightgbm as lgb

    df = build(Path(a.root), Path(a.cache), Path(a.alfa_words))
    feats = [c for c in df.columns if c not in ("rid", "split", "level", "y")]
    tr, te = df[df.split.isin(["train", "val"])], df[df.split.isin(["test", "test_passage"])]
    print(f"{df.rid.nunique()} answers, {len(df)} words (train {len(tr)}, held-out {len(te)})")
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1)
    m = lgb.LGBMClassifier(**params).fit(tr[feats], tr.y)
    p, y = m.predict(te[feats]), te.y.to_numpy()
    base = np.where(te.status != 0, 2, 0)
    res = {"agreement": float(np.mean(p == y)), "baseline_read_is_correct": float(np.mean(base == y)),
           "correct_vs_not": float(np.mean((p == 0) == (y == 0))), "n_words": int(len(y))}
    for lo, hi in ((0, 43), (43, 66), (66, 91)):
        mk = ((te.level >= lo) & (te.level < hi)).to_numpy()
        res[f"agreement_level_{lo}_{hi}"] = float(np.mean(p[mk] == y[mk])) if mk.any() else None
    print(json.dumps(res, indent=1))
    print(pd.crosstab(pd.Series([W.LABELS[i] for i in y], name="alfa"),
                      pd.Series([W.LABELS[i] for i in p], name="ours")))

    final = lgb.LGBMClassifier(**params).fit(df[feats], df.y)   # all answers for the saved model
    W.CLASSIFIER.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": feats, "metrics": res}, W.CLASSIFIER)
    print("saved", W.CLASSIFIER)


if __name__ == "__main__":
    main()
