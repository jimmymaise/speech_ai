"""Turn features into trait scores (trained model if available, rules otherwise) + learner feedback."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import TRAITS
from . import model as model_lib
from .levels import estimate, to_pte_scale
from .words import marked_pauses, tagged_answer, word_report

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "read_aloud.joblib"


def _clip(x: float) -> float:
    return float(min(5.0, max(0.0, x)))


def _g(f: dict, k: str, default: float = 0.0) -> float:
    v = f.get(k, default)
    return default if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


# ----------------------------------------------------------------- rule-based
def rule_scores(f: dict) -> dict[str, float]:
    """Transparent fallback rules. Used for traits without a trained model."""
    if _g(f, "no_speech") >= 1:
        return {t: 0.0 for t in TRAITS}

    # Content: PTE counts every omission, insertion and substitution. Scaled so that
    # ~40% of the passage wrong -> 0.
    n = max(_g(f, "n_ref_words", 1), 1)
    errors = _g(f, "n_errors")
    content = _clip(5.0 * (1.0 - errors / (0.4 * n)))

    # Oral fluency: start at 5, subtract for disfluencies and unnatural pace
    flu = 5.0
    flu -= 0.6 * _g(f, "n_bad_pauses")            # pauses not at punctuation
    flu -= 0.5 * _g(f, "n_pauses_1")              # any pause >= 1 s
    flu -= 0.4 * (_g(f, "n_fillers") + _g(f, "n_repetitions"))
    rate = _g(f, "speech_rate_syl_s", 3.5)
    if rate < 3.2:
        flu -= 1.5 * (3.2 - rate)
    elif rate > 6.0:
        flu -= 1.0 * (rate - 6.0)
    if _g(f, "f0_std_st", 3.0) < 1.5:              # monotone
        flu -= 0.5
    flu -= min(2.0, max(0.0, _g(f, "initial_silence_s") - 2.0))
    fluency = _clip(flu)

    # Pronunciation: share of phones that look wrong + phone recogniser disagreement
    pron = 5.0 - 6.0 * _g(f, "pct_phones_bad_25") - 3.0 * max(0.0, _g(f, "phone_error_rate") - 0.12)
    pronunciation = _clip(pron)

    # PTE: if content is very low, the other traits cannot be scored highly
    if content < 1:
        fluency, pronunciation = min(fluency, 1.0), min(pronunciation, 1.0)
    return {"content": content, "fluency": fluency, "pronunciation": pronunciation}


# -------------------------------------------------------------------- scorer
class Scorer:
    def __init__(self, model_path: str | Path | None = DEFAULT_MODEL_PATH):
        self.bundle = model_lib.load(model_path) if model_path else None

    def score(self, analysis: dict, pte_type: str = "academic") -> dict:
        """pte_type: "academic" or "core" (used by models trained with pte_type as a feature)."""
        f = {**analysis["features"], "pte_type": model_lib.CONTEXT_FEATURES["pte_type"][pte_type]}
        scores = rule_scores(f)
        source = {t: "rules" for t in TRAITS}
        on_90 = {t: to_pte_scale(v, 5.0) for t, v in scores.items()}
        if self.bundle and _g(f, "no_speech") < 1:
            row = pd.DataFrame([{c: f.get(c, np.nan) for c in self.bundle["features"]}], dtype=float)
            for t, m in self.bundle["models"].items():
                hi = self.bundle.get("scale", {}).get(t, 5.0)
                # report on the PTE 0-5 trait scale; the raw model-scale value is kept too
                raw = min(hi, max(0.0, float(m.predict(row)[0])))
                scores[t] = _clip(raw * 5.0 / hi)
                on_90[t] = to_pte_scale(raw, hi)
                source[t] = "model" if hi == 5.0 else f"model (raw {raw:.0f}/{hi:g})"
        if scores["content"] < 1:
            for t in ("fluency", "pronunciation"):
                scores[t] = min(scores[t], 1.0)
                on_90[t] = min(on_90[t], to_pte_scale(1.0, 5.0))
        words = word_report(analysis.get("details", {}), features=analysis["features"],
                            pte_type=model_lib.CONTEXT_FEATURES["pte_type"][pte_type])
        return {
            "scores": {t: round(v, 2) for t, v in scores.items()},
            "scores_rounded": {t: int(round(v)) for t, v in scores.items()},
            "score_source": source,
            "estimate": estimate(on_90),
            "alfa_style": alfa_style(on_90, words, analysis, pte_type),
            "words": words,
            "feedback": feedback(analysis, words),
        }


# Alfa's Read Aloud speaking score is a weighted mix of fluency and pronunciation (content goes to
# Reading). Least-squares fit on 206k Alfa answers, mean error < 0.4 points.
SPEAKING_WEIGHTS = {"academic": (0.014, 0.405, 0.583, -0.172), "core": (0.005, 0.600, 0.396, -0.095)}
# Alfa's "x out of 5" pronunciation is a banding of its 0-90 pronunciation (most common value per score)
PRON_5_BANDS = [(88, 5.0), (84, 4.5), (76, 4.0), (65, 3.5), (55, 3.0), (35, 2.5), (7, 2.0), (1, 1.5), (0, 0.0)]


def pron_out_of_5(p90: float) -> float:
    return next(v for lo, v in PRON_5_BANDS if p90 >= lo)


def alfa_style(on_90: dict, words: list[dict], analysis: dict, pte_type: str) -> dict:
    """Same fields as an Alfa Read Aloud result. Scores are 0-90 (silence -> 0, like Alfa)."""
    no_speech = _g(analysis["features"], "no_speech") >= 1
    s = {t: 0 if no_speech else int(round(v)) for t, v in on_90.items()}
    c, f, p, b = SPEAKING_WEIGHTS[pte_type]
    speaking = 0 if no_speech else int(round(min(90, max(10, c * s["content"] + f * s["fluency"]
                                                       + p * s["pronunciation"] + b))))
    pauses = marked_pauses(analysis.get("details", {}))
    n = {lab: sum(w["label"] == lab for w in words) for lab in ("CORRECT", "AVERAGE", "WRONG")}
    return {
        "speech_content": s["content"],
        "speech_fluency": s["fluency"],
        "display_pronunciation": s["pronunciation"],
        "speech_pronunciation": f"{pron_out_of_5(s['pronunciation']):g} out of 5",
        "speakingScore": speaking,
        "userAnswer": tagged_answer(words, pauses),
        "answerCompare": {"pause_count": len(pauses), "bad_word_count": f"{n['WRONG']} words",
                          "average_word_count": f"{n['AVERAGE']} words",
                          "total_word_count": f"{len(words)} words", "good_word_count": f"{n['CORRECT']} words"},
        "transcript": analysis.get("details", {}).get("transcript", ""),
    }


def feedback(analysis: dict, words: list[dict] | None = None) -> dict:
    f, d = analysis["features"], analysis.get("details", {})
    if _g(f, "no_speech") >= 1:
        return {"summary": [d.get("note", "No speech detected.")]}
    ops = d.get("ops", [])
    omitted = [o["ref"] for o in ops if o["op"] == "del"]
    inserted = [o["said"] for o in ops if o["op"] == "ins"]
    substituted = [{"expected": o["ref"], "said": o["said"]} for o in ops if o["op"] == "sub"]
    bad_pauses = [p for p in d.get("pauses", []) if not p["at_punctuation"]]

    tips = []
    if omitted:
        tips.append(f"Skipped {len(omitted)} word(s): read every word, even if you stumble.")
    if _g(f, "pct_passage_covered", 1) < 0.9:
        tips.append("You did not finish the passage: keep a steady pace so you reach the end.")
    if bad_pauses:
        tips.append(f"{len(bad_pauses)} pause(s) in the middle of a phrase: pause only at commas and full stops.")
    if _g(f, "n_fillers") + _g(f, "n_repetitions") > 0:
        tips.append("Avoid 'uh/um' and repeating words: if you make a mistake, keep going.")
    rate = _g(f, "speech_rate_syl_s", 3.5)
    if rate < 3.0:
        tips.append(f"Speaking rate is slow ({rate:.1f} syllables/s): aim for about 3.5-4.5.")
    elif rate > 6.0:
        tips.append(f"Speaking rate is very fast ({rate:.1f} syllables/s): slow down slightly.")
    if _g(f, "initial_silence_s") > 2.0:
        tips.append("Start speaking sooner: in PTE the microphone closes after 3 s of silence.")
    words = words or []
    endings = [w for w in words if any(i.get("ending_consonant") for i in w["issues"])]
    dropped = [w["word"] for w in words if any(i.get("ending_consonant") and i["problem"] == "missing"
                                                for i in w["issues"])]
    if dropped:
        tips.append(f"Ending sounds dropped in {len(dropped)} word(s) (e.g. {', '.join(dropped[:4])}): "
                    "finish each word, especially /t/, /d/, /s/, /z/.")
    elif len(endings) >= 3 or _g(f, "pct_final_cons_bad") > 0.2:
        tips.append("Final consonants are often weak (e.g. /t/, /s/, /z/): finish each word clearly.")
    if _g(f, "pct_th_dh_bad") > 0.3:
        tips.append("Practise 'th' sounds /θ/ /ð/: tongue between the teeth, not /t/ /d/.")
    if _g(f, "f0_std_st", 3.0) < 1.5:
        tips.append("Intonation is flat: stress key words and let the pitch fall at the end of sentences.")

    return {
        "summary": tips,
        "omitted_words": omitted,
        "inserted_words": inserted,
        "substituted_words": substituted,
        "mispronounced_words": [{"word": w["word"], "index": w["index"], "start": w.get("start"),
                                 "label": w["label"], "issues": w["issues"], "message": w["message"]}
                                for w in words if w["issues"]],
        "dropped_ending_sounds": dropped,
        "bad_pauses": bad_pauses,
        "transcript": d.get("transcript", ""),
    }
