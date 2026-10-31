"""End-to-end analysis: audio + passage -> flat feature dict + human-readable details."""
from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .asr import DEFAULT_MODEL, transcribe
from .audio import audio_quality, load_audio
from .content import content_features
from .fluency import fluency_features
from .phonemes import (FINAL_CONSONANTS, PhoneModel, is_vowel, phone_error_rate, prefetch_phones,
                       score_phones, word_phones)
from .text import FILLERS, tokenize_hypothesis, tokenize_reference

# GOP below this marks a phone as probably wrong. Starting point; tune on labelled data.
BAD_PHONE_GOP = -2.5


class Analyzer:
    def __init__(self, asr_model: str = DEFAULT_MODEL, device: str | None = None):
        self.asr_model = asr_model
        self.pm = PhoneModel(device)

    def analyze(self, audio: str | np.ndarray, passage: str) -> dict:
        out = self._analyze(audio, passage)
        out["features"] = {k: float(v) for k, v in out["features"].items()}
        return out

    def _analyze(self, audio: str | np.ndarray, passage: str) -> dict:
        wav = load_audio(audio) if isinstance(audio, str) else audio.astype(np.float32)
        ref = tokenize_reference(passage)
        ref_words = [r.text for r in ref]
        feats: dict = {f"aq_{k}": v for k, v in audio_quality(wav).items()}

        if wav.size < 0.3 * SAMPLE_RATE:
            return _empty(feats, ref_words, "audio too short")

        asr_text, asr_words = transcribe(wav, self.asr_model)
        hyp = tokenize_hypothesis(" ".join(w.word for w in asr_words) or asr_text)
        feats["asr_mean_word_prob"] = float(np.mean([w.prob for w in asr_words])) if asr_words else 0.0
        if not [h for h in hyp if h not in FILLERS]:
            return _empty(feats, ref_words, "no speech recognised", asr_text)

        cfeats, ops = content_features(ref_words, hyp)
        feats.update(cfeats)

        # word sequence actually spoken, in order: passage words read (+ substitutions) and insertions
        spoken = []
        for o in ops:
            if o.op == "match":
                spoken.append((ref_words[o.ref_idx], o.ref_idx, "match"))
            elif o.op == "sub":
                spoken.append((hyp[o.hyp_idx], o.ref_idx, "sub"))  # align what was actually said
            elif o.op == "ins":
                spoken.append((hyp[o.hyp_idx], None, "ins"))
        prefetch_phones([w for w, _, _ in spoken])
        spoken = [s for s in spoken if word_phones(s[0])]

        lp = self.pm.log_probs(wav)
        res = score_phones(self.pm, lp, [(w, r) for w, r, _ in spoken])
        details = {"transcript": asr_text, "ops": [], "mispronounced": [], "pauses": [],
                   "ref_words": [r.raw for r in ref]}
        for o in ops:
            details["ops"].append({"op": o.op, "ref_idx": o.ref_idx,
                                   "ref": ref[o.ref_idx].raw.strip(".,;:!?\"'()") if o.ref_idx is not None else None,
                                   "said": hyp[o.hyp_idx] if o.hyp_idx is not None else None})
        if res is None:
            feats["alignment_failed"] = 1.0
            return {"features": feats, "details": details}
        feats["alignment_failed"] = 0.0
        aligned, phones = res
        # `aligned` is index-aligned with `spoken` (every spoken word has phones)
        # ---- pronunciation: only passage words that were read as the right word
        matched_words = [w for w, s in zip(aligned, spoken) if s[2] == "match"]
        feats.update(_pron_features(matched_words))
        canon = [p.phone for w in aligned for p in w.phones]
        feats["phone_error_rate"] = phone_error_rate(canon, self.pm.greedy_decode(lp))

        for w in matched_words:
            worst = min(w.phones, key=lambda p: p.gop)
            if worst.gop < BAD_PHONE_GOP:
                details["mispronounced"].append({
                    "word": ref[w.ref_idx].raw.strip(".,;:!?\"'()"), "ref_idx": w.ref_idx,
                    "expected": worst.phone, "heard": worst.competitor, "gop": round(worst.gop, 2),
                    "start": round(float(w.start), 2)})

        # ---- fluency
        ffeats, pauses = fluency_features(wav, aligned, {i: r.break_after for i, r in enumerate(ref)},
                                          cfeats["n_fillers"], cfeats["n_repetitions"])
        feats.update(ffeats)
        details["pauses"] = pauses
        details["words"] = [{"word": w.word, "ref_idx": w.ref_idx, "op": s[2], "start": round(float(w.start), 2),
                             "end": round(float(w.end), 2),
                             "gop": round(float(np.mean([p.gop for p in w.phones])), 2),
                             # [phone, gop, most likely other phone, max posterior of phone nearby,
                             #  max posterior of the other phone] for word-level feedback
                             "phones": [[p.phone, round(float(p.gop), 2), p.competitor, round(float(p.pmax), 3),
                                         round(float(p.comp_pmax), 3)]
                                        for p in w.phones]}
                            for w, s in zip(aligned, spoken)]
        return {"features": feats, "details": details}


def _pron_features(words) -> dict:
    ph = [p for w in words for p in w.phones]
    if not ph:
        return {"gop_mean": np.nan}
    g = np.array([p.gop for p in ph])
    vow = np.array([p.gop for p in ph if is_vowel(p.phone)])
    con = np.array([p.gop for p in ph if not is_vowel(p.phone)])
    finals = np.array([w.phones[-1].gop for w in words if w.phones[-1].phone in FINAL_CONSONANTS])
    word_min = np.array([min(p.gop for p in w.phones) for w in words])
    word_mean = np.array([np.mean([p.gop for p in w.phones]) for w in words])

    def m(a):
        return float(a.mean()) if a.size else np.nan

    return {
        "gop_mean": float(g.mean()),
        "gop_median": float(np.median(g)),
        "gop_p10": float(np.percentile(g, 10)),
        "gop_std": float(g.std()),
        "lpp_mean": float(np.mean([p.lpp for p in ph])),
        "gop_vowel_mean": m(vow),
        "gop_consonant_mean": m(con),
        "gop_final_cons_mean": m(finals),
        "pct_final_cons_bad": float((finals < BAD_PHONE_GOP).mean()) if finals.size else np.nan,
        "pct_phones_bad_1": float((g < -1.0).mean()),
        "pct_phones_bad_25": float((g < -2.5).mean()),
        "pct_phones_bad_5": float((g < -5.0).mean()),
        "pct_words_bad": float((word_min < BAD_PHONE_GOP).mean()),
        "word_gop_mean_p10": float(np.percentile(word_mean, 10)),
        "pct_th_dh_bad": float(np.mean([p.gop < BAD_PHONE_GOP for p in ph if p.phone in ("θ", "ð")]))
        if any(p.phone in ("θ", "ð") for p in ph) else np.nan,
    }


def _empty(feats: dict, ref_words: list[str], reason: str, transcript: str = "") -> dict:
    feats.update({"no_speech": 1.0, "n_ref_words": len(ref_words), "pct_match": 0.0,
                  "pct_omit": 1.0, "pct_passage_covered": 0.0, "wer": 1.0})
    return {"features": feats, "details": {"transcript": transcript, "note": reason, "ref_words": ref_words}}
