"""Word-level feedback: a CORRECT / AVERAGE / WRONG label for every passage word, the sound problems
inside each word (incl. dropped ending sounds), and an Alfa-style tagged answer string.

Works on `analysis["details"]` only, so it can be re-run on cached analyses after re-tuning.
Thresholds are tuned against Alfa's word labels (scripts/tune_word_labels.py writes word_labels.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .phonemes import FINAL_CONSONANTS, is_vowel

TUNED = Path(__file__).resolve().parent.parent / "models" / "word_labels.json"
CLASSIFIER = Path(__file__).resolve().parent.parent / "models" / "word_labels.joblib"
LABELS = ["CORRECT", "AVERAGE", "WRONG"]
# answer-level features the word classifier also sees (how the whole answer went)
ANSWER_FEATURES = ["pct_match", "pct_omit", "pct_passage_covered", "gop_mean", "phone_error_rate",
                   "asr_mean_word_prob", "speech_rate_syl_s"]
_clf_cache: dict = {}
DEFAULTS = {
    "average_gop": -2.5,     # worst sound below this -> AVERAGE
    "wrong_gop": -5.0,       # worst sound below this -> WRONG
    "missing_pmax": 0.05,    # sound never heard nearby ...
    "missing_comp_pmax": 0.3,  # ... and no other sound clearly in its place -> "missing"
    "pause_min_s": 0.5,      # shorter mid-phrase silences are not marked
    "pause_slash_s": 0.5,    # one "/" per this many seconds of pause
    "pause_include_punctuation": False,  # also mark pauses at commas / full stops
}

# Small words that are normally said in a weak form: "a" /ə/, "the" /ðə/, "to" /tə/, "and" /ən/ ...
# A full vowel reduced to a schwa-like vowel in these words is correct English, not an error.
FUNCTION_WORDS = {
    "a", "an", "the", "and", "to", "of", "for", "from", "at", "as", "was", "were", "are", "can", "could",
    "would", "should", "has", "have", "had", "do", "does", "but", "or", "nor", "than", "that", "them",
    "us", "some", "her", "his", "him", "your", "you", "be", "been", "is", "into", "onto", "by", "there",
    "their", "our", "must", "will", "shall", "am", "in", "on", "with", "its", "it",
}
REDUCED = {"ə", "ɐ", "ʌ", "ɪ", "ᵻ", "ɚ", "ɜ", "ɜː", "ʊ", "i", "u", "ɛ", "ɑ", "ɑː", "ɔ"}
DROPPABLE_IN = {"and": {"d"}, "just": {"t"}, "must": {"t"}, "next": {"t"}, "first": {"t"}}


def _cfg() -> dict:
    cfg = dict(DEFAULTS)
    if TUNED.exists():
        cfg.update(json.loads(TUNED.read_text()))
    return cfg


def _excused(word: str, phone: str, competitor: str, is_last: bool) -> bool:
    w = word.lower()
    if w in FUNCTION_WORDS and is_vowel(phone) and competitor in REDUCED:
        return True
    return is_last and phone in DROPPABLE_IN.get(w, ())


def sound_issues(word: str, phones: list, cfg: dict) -> list[dict]:
    """phones: [[phone, gop, competitor, pmax, comp_pmax], ...] as stored in details["words"]."""
    out = []
    n = len(phones)
    for i, ph in enumerate(phones):
        phone, gop, comp = ph[0], ph[1], ph[2]
        pmax = ph[3] if len(ph) > 3 else 1.0
        comp_pmax = ph[4] if len(ph) > 4 else 1.0
        if gop >= cfg["average_gop"] or _excused(word, phone, comp, i == n - 1):
            continue
        pos = "ending" if i == n - 1 else ("start" if i == 0 else "middle")
        missing = pmax < cfg["missing_pmax"] and comp_pmax < cfg["missing_comp_pmax"]
        issue = {"sound": phone, "position": pos, "gop": gop,
                 "problem": "missing" if missing else "replaced"}
        if not missing:
            issue["heard"] = comp
        if pos == "ending" and phone in FINAL_CONSONANTS:
            issue["ending_consonant"] = True
        out.append(issue)
    return out


def _message(w: dict) -> str | None:
    if w["status"] == "omitted":
        return "not read"
    if w["status"] == "substituted":
        return f"read as '{w['said']}'"
    msgs = []
    for s in w["issues"]:
        where = {"ending": "ending sound", "start": "first sound", "middle": "sound"}[s["position"]]
        if s["problem"] == "missing":
            msgs.append(f"{where} /{s['sound']}/ missing")
        else:
            msgs.append(f"{where} /{s['sound']}/ sounded like /{s['heard']}/")
    return "; ".join(msgs) or None


def _classifier():
    if "m" not in _clf_cache:
        import joblib
        _clf_cache["m"] = joblib.load(CLASSIFIER) if CLASSIFIER.exists() else None
    return _clf_cache["m"]


def word_feature_rows(details: dict, features: dict, pte_type: int) -> list[dict]:
    """Model inputs for each passage word (same order as word_report)."""
    loose = {**_cfg(), "average_gop": 0.0}
    ours = word_report(details, loose, use_classifier=False)
    timed = {w["ref_idx"]: w for w in details.get("words", []) if w.get("ref_idx") is not None}
    status = {"read": 0, "substituted": 1, "omitted": 2}
    n = len(ours)
    rows = []
    for o in ours:
        t = timed.get(o["index"]) if o["status"] == "read" else None
        ph = t["phones"] if t and "phones" in t else []
        g = [p[1] for p in ph]
        rows.append({
            "status": status[o["status"]], "worst": min((x["gop"] for x in o["issues"]), default=0.0),
            "raw_min": min(g, default=np.nan), "mean_gop": float(np.mean(g)) if g else np.nan,
            "n_issues": len(o["issues"]), "n_ph": len(ph),
            "dur": (t["end"] - t["start"]) if t else np.nan,
            "min_pmax": min((p[3] for p in ph if len(p) > 3), default=np.nan),
            "fn": int(o["word"].lower() in FUNCTION_WORDS), "pos": o["index"] / max(n, 1), "wlen": len(o["word"]),
            "prev_status": int(o["index"] > 0 and ours[o["index"] - 1]["status"] != "read"),
            "next_status": int(o["index"] < n - 1 and ours[o["index"] + 1]["status"] != "read"),
            **{k: features.get(k, np.nan) for k in ANSWER_FEATURES}, "ptype": pte_type,
        })
    return rows


def word_report(details: dict, cfg: dict | None = None, features: dict | None = None,
                pte_type: int = 1, use_classifier: bool = True) -> list[dict]:
    """One entry per passage word, in passage order. With a trained word classifier and the answer's
    features, the label is the classifier's (trained on Alfa word labels); otherwise GOP thresholds."""
    cfg = cfg or _cfg()
    ref = details.get("ref_words") or []
    if not ref:
        return []
    by_ref, k = {}, 0
    for o in details.get("ops", []):   # ops walk the passage in order; only insertions skip it
        if o["op"] == "ins":
            continue
        by_ref[o["ref_idx"] if o.get("ref_idx") is not None else k] = o
        k += 1
    timed = {w["ref_idx"]: w for w in details.get("words", []) if w.get("ref_idx") is not None}
    out = []
    for i, raw in enumerate(ref):
        o = by_ref.get(i, {"op": "del"})
        w = {"index": i, "word": raw.strip(".,;:!?\"'()"), "issues": []}
        t = timed.get(i)
        if t:
            w["start"], w["end"] = t["start"], t["end"]
        if o["op"] == "del":
            w.update(label="WRONG", status="omitted")
        elif o["op"] == "sub":
            w.update(label="WRONG", status="substituted", said=o.get("said"))
        else:
            w["status"] = "read"
            w["issues"] = sound_issues(w["word"], t["phones"], cfg) if t and "phones" in t else []
            worst = min((s["gop"] for s in w["issues"]), default=0.0)
            w["label"] = ("WRONG" if worst < cfg["wrong_gop"] else
                          "AVERAGE" if worst < cfg["average_gop"] else "CORRECT")
        out.append(w)

    clf = _classifier() if use_classifier and features is not None else None
    if clf is not None and out:
        import pandas as pd
        X = pd.DataFrame(word_feature_rows(details, features, pte_type))[clf["features"]]
        for w, k in zip(out, clf["model"].predict(X)):
            if w["status"] == "read":
                w["label"] = LABELS[int(k)]
    for w in out:
        if w["label"] == "CORRECT":
            w["issues"] = []          # don't show sound problems on a word judged correct
        w["message"] = _message(w) or ("not clear" if w["label"] != "CORRECT" else None)
    return out


def marked_pauses(details: dict, cfg: dict | None = None) -> dict[int, int]:
    """{ref word index the pause follows: number of '/'} for pauses in the middle of a phrase."""
    cfg = cfg or _cfg()
    out = {}
    for p in details.get("pauses", []):
        if ((p.get("at_punctuation") and not cfg["pause_include_punctuation"])
                or p.get("after_ref_idx") is None or p["dur"] < cfg["pause_min_s"]):
            continue
        out[p["after_ref_idx"]] = max(1, round(p["dur"] / cfg["pause_slash_s"]))
    return out


def tagged_answer(words: list[dict], pauses: dict[int, int]) -> str:
    """Alfa's format: [CORRECT]runs of words[/CORRECT] [[AVERAGE]]..[[/AVERAGE]] [[WRONG]]..[[/WRONG]]
    [[PAUSE]]/[[/PAUSE]]"""
    parts, run, run_label = [], [], None

    def flush():
        if run:
            lab = run_label
            parts.append(f"[{lab}]{' '.join(run)}[/{lab}]" if lab == "CORRECT"
                         else f"[[{lab}]]{' '.join(run)}[[/{lab}]]")
            run.clear()

    for w in words:
        if w["label"] != run_label:
            flush()
            run_label = w["label"]
        run.append(w["word"].lower())
        if w["index"] in pauses:
            flush()
            parts.append(f"[[PAUSE]]{'/' * pauses[w['index']]}[[/PAUSE]]")
    flush()
    return " ".join(parts)
