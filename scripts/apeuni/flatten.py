"""Flatten APEUni answers of one task into a parquet table (one row per answer).

    python scripts/apeuni/flatten.py <task>
    task: repeat_sentence | describe_image | retell_lecture | summarize_group_discussion | respond_situation
    (read_aloud: scripts/apeuni_formulas.py)
"""
import sys

import numpy as np
import pandas as pd

from common import out_path, questions


def row_open(q, qj, a):
    """Describe Image, Retell Lecture, Summarize Group Discussion: words with timings and sound scores."""
    c = a["scores"]["components"]
    sd = a.get("score_details") or {}
    det = sd.get("details") or []
    cw = [t for t in det if t.get("type") == "content_word"]
    dur = np.array([(t["end_pos"] - t["beg_pos"]) / 100 for t in cw if "end_pos" in t])
    ph = np.array([s["score"] for t in cw for s in t.get("syllables", [])])
    rec = a.get("recognized_text") or ""
    return dict(
        n_words=len(cw), rec_words=len(rec.split()), letters=sum(ch.isalpha() for ch in rec),
        seconds=dur.sum() if len(dur) else np.nan, n_long1=int((dur > 1).sum()) if len(dur) else 0,
        d_std=dur.std() if len(dur) else np.nan, sound_mean=ph.mean() if len(ph) else np.nan,
        word_mean=np.mean([float(t["score"]) for t in cw]) if cw else np.nan,
        xP=(sd.get("extra_details") or {}).get("pronunciation_score"),
        ccomment=(c["content"].get("comment") or "")[:70], rec=rec,
        ref=qj.get("prediction_text") or qj.get("answer_in_text") or "")


def row_rts(q, qj, a):
    """Respond to a Situation: words carry a 0-100 score and a 0-1 tc_fluency, no timings."""
    sd = a["score_details"]
    det = sd["details"]
    ws = np.array([float(t["score"]) for t in det]) if det else np.array([np.nan])
    tf = np.array([float(t.get("tc_fluency", np.nan)) for t in det]) if det else np.array([np.nan])
    return dict(n_words=len(det), word_mean=np.nanmean(ws), tcf_mean=np.nanmean(tf), tcf_low=float((tf < 0.5).mean()),
                tc_pronun=sd["extra_details"].get("tc_pronun"), tc_fluency=sd["extra_details"].get("tc_fluency"),
                rec=" ".join(t["target"] for t in det), ref=qj.get("prediction_text") or "")


def row_rs(q, qj, a):
    """Repeat Sentence: one entry per sentence word: content_word / omission / incorrect_order."""
    det = a["score_details"]["details"]
    cw = [t for t in det if t["type"] == "content_word"]
    ws = np.array([float(t["score"]) for t in cw]) if cw else np.array([np.nan])
    text = qj["prediction_text"]
    return dict(n=len(det), letters=sum(ch.isalpha() for ch in text), n_cw=len(cw),
                n_om=sum(t["type"] == "omission" for t in det), n_ord=sum(t["type"] == "incorrect_order" for t in det),
                word_mean=np.nanmean(ws), rec=a.get("recognized_text") or "", ref=text,
                missed=" ".join(t["target"] for t in det if t["type"] != "content_word"))


ROW = {"respond_situation": row_rts, "repeat_sentence": row_rs}


def main():
    task = sys.argv[1]
    make = ROW.get(task, row_open)
    rows = []
    for q, qj, answers in questions(task):
        for x in answers:
            a = x["attributes"]
            c = a["scores"]["components"]
            rows.append(dict(q=q, id=int(a["id"]), total=float(a["total_score"]), C=float(c["content"]["score"]),
                             P=float(c["pronunciation"]["score"]), F=float(c["fluency"]["score"]),
                             engine=(a.get("algo_version_tag") or {}).get("tag_name"), t=a.get("timetaken"),
                             **make(q, qj, a)))
    df = pd.DataFrame(rows).drop_duplicates("id")
    path = out_path(f"{task}.parquet")
    df.to_parquet(path)
    print(f"{task}: {len(df)} answers, {df.q.nunique()} questions -> {path}")


if __name__ == "__main__":
    main()
