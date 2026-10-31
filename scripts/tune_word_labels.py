"""Tune the CORRECT / AVERAGE / WRONG thresholds and pause marks to agree with Alfa's word labels.

  python scripts/tune_word_labels.py data/alfa_r1 cache/alfa_r1 [--split train] [--write]

Uses cached analyses that contain per-phone details (extracted after the word-feedback change).
Thresholds are tuned on --split (default train) and reported on test + test_passage.
--write saves models/word_labels.json, which pte_scorer.words loads.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pte_scorer import words as W  # noqa: E402

LABELS = ["CORRECT", "AVERAGE", "WRONG"]


def norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalnum() or c == "'")


def collect(root: Path, cache: Path, alfa_words_path: Path):
    lab = pd.read_csv(root / "labels.csv")
    ids = set(lab.response_id)
    split = dict(zip(lab.response_id, lab.split))
    alfa = {}
    with open(alfa_words_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["response_id"] in ids:
                alfa[r["response_id"]] = r["words"]
    rows, pause_rows = [], []
    loose = {**W._cfg(), "average_gop": 0.0}   # keep every non-excused sound, threshold later
    for rid in ids:
        f = cache / f"{rid}.json"
        if not f.exists() or rid not in alfa:
            continue
        d = json.loads(f.read_text())["details"]
        if not d.get("ref_words") or not any("phones" in w for w in d.get("words", [])):
            continue
        ours = W.word_report(d, loose)
        aw = [w for w in alfa[rid] if w[1] != "PAUSE"]
        # Alfa pause marks: index (in its word list) of the word before each mark
        k, alfa_pause_after = -1, {}
        for w in alfa[rid]:
            if w[1] == "PAUSE":
                alfa_pause_after[k] = len(w[0])
            else:
                k += 1
        sm = SequenceMatcher(None, [norm(w["word"]) for w in ours], [norm(w[0]) for w in aw], autojunk=False)
        a2o = {}
        for blk in sm.get_matching_blocks():
            for j in range(blk.size):
                o = ours[blk.a + j]
                a2o[blk.b + j] = blk.a + j
                worst = min((s["gop"] for s in o["issues"]), default=0.0)
                rows.append({"rid": rid, "split": split[rid], "alfa": aw[blk.b + j][1], "status": o["status"],
                             "worst": worst, "n_issues": len(o["issues"])})
        alfa_p = {a2o[a]: n for a, n in alfa_pause_after.items() if a in a2o}
        for p in d.get("pauses", []):
            if p.get("after_ref_idx") is None:
                continue
            pause_rows.append({"split": split[rid], "dur": p["dur"], "punct": p["at_punctuation"],
                               "alfa_marked": p["after_ref_idx"] in alfa_p,
                               "alfa_slashes": alfa_p.get(p["after_ref_idx"], 0)})
        pause_rows.append({"split": split[rid], "dur": np.nan, "punct": None, "alfa_marked": None,
                           "alfa_total": len(alfa_pause_after)})
    return pd.DataFrame(rows), pd.DataFrame(pause_rows)


def predict(df, avg, wrong):
    lab = np.where(df.worst < wrong, "WRONG", np.where(df.worst < avg, "AVERAGE", "CORRECT"))
    return np.where(df.status != "read", "WRONG", lab)


def macro_f1(y, p):
    f = []
    for c in LABELS:
        tp = np.sum((y == c) & (p == c))
        prec, rec = tp / max(np.sum(p == c), 1), tp / max(np.sum(y == c), 1)
        f.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("cache")
    ap.add_argument("--alfa-words", default="data/alfa/word_labels.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    df, pz = collect(Path(a.root), Path(a.cache), Path(a.alfa_words))
    tr, te = df[df.split == a.split], df[df.split.isin(["test", "test_passage"])]
    print(f"{df.rid.nunique()} answers, {len(df)} words (tune {len(tr)}, report {len(te)})")
    print("Alfa label share:", (df.alfa.value_counts(normalize=True).round(3)).to_dict())

    best = None
    for avg in np.arange(-0.5, -6.01, -0.25):
        for wrong in np.arange(avg - 0.25, -12.01, -0.25):
            s = macro_f1(tr.alfa.to_numpy(), predict(tr, avg, wrong))
            if best is None or s > best[0]:
                best = (s, round(float(avg), 2), round(float(wrong), 2))
    _, avg, wrong = best
    for name, d in (("tune", tr), ("held-out", te)):
        if len(d) == 0:
            continue
        p = predict(d, avg, wrong)
        y = d.alfa.to_numpy()
        print(f"\n[{name}] average_gop={avg} wrong_gop={wrong}: agreement {np.mean(p == y):.1%}, "
              f"macro-F1 {macro_f1(y, p):.3f}, CORRECT-vs-not agreement {np.mean((p == 'CORRECT') == (y == 'CORRECT')):.1%}")
        print(pd.crosstab(pd.Series(y, name="alfa"), pd.Series(p, name="ours")).reindex(index=LABELS, columns=LABELS))

    # pauses: which of our pauses Alfa marks, and how many slashes per second
    pp = pz[pz.dur.notna() & (pz.split == a.split)]
    best_p = None
    for punct_too in (False, True):
        for mn in np.arange(0.25, 1.51, 0.05):
            pred = (pp.dur >= mn) & (punct_too | ~pp.punct.astype(bool))
            y = pp.alfa_marked.astype(bool)
            tp = (pred & y).sum()
            f1 = 2 * tp / max(pred.sum() + y.sum(), 1)
            if best_p is None or f1 > best_p[0]:
                best_p = (f1, punct_too, round(float(mn), 2))
    m = pp[pp.alfa_marked.astype(bool)]
    slash_s = float(np.median(m.dur / m.alfa_slashes.clip(lower=1))) if len(m) else W.DEFAULTS["pause_slash_s"]
    print(f"\npauses: best F1 {best_p[0]:.3f} with min {best_p[2]} s, include punctuation pauses: {best_p[1]}; "
          f"~{slash_s:.2f} s per '/'")
    print(f"Alfa marks per answer {pz.alfa_total.mean():.2f};  our pauses >= min: "
          f"{(pp.dur >= best_p[2]).sum() / max(pz[pz.split == a.split].alfa_total.notna().sum(), 1):.2f}")

    if a.write:
        out = {"average_gop": avg, "wrong_gop": wrong, "pause_min_s": best_p[2],
               "pause_include_punctuation": bool(best_p[1]), "pause_slash_s": round(slash_s, 2),
               "tuned_on_words": int(len(tr))}
        W.TUNED.parent.mkdir(exist_ok=True)
        W.TUNED.write_text(json.dumps(out, indent=2))
        print("wrote", W.TUNED)


if __name__ == "__main__":
    main()
