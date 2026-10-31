"""Oral fluency & prosody features: pauses, rate, rhythm, intonation."""
from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE
from .phonemes import WordAlignment, is_vowel

HOP_S = 0.01


def speech_mask(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Energy-based voice activity per 10 ms frame, threshold relative to the loud part of the clip."""
    import librosa

    if wav.size < sr * 0.05:
        return np.zeros(0, bool)
    rms = librosa.feature.rms(y=wav, frame_length=int(0.03 * sr), hop_length=int(HOP_S * sr))[0]
    db = 20 * np.log10(rms + 1e-9)
    loud = np.percentile(db, 95)
    noise = np.percentile(db, 10)
    thr = max(noise + 0.35 * (loud - noise), loud - 40)
    mask = db > thr
    # close tiny gaps (< 60 ms) inside speech, e.g. stop closures
    gap = 0
    for i in range(len(mask)):
        if mask[i]:
            if 0 < gap <= 6:
                mask[i - gap:i] = True
            gap = 0
        else:
            gap += 1
    return mask


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def fluency_features(wav: np.ndarray, words: list[WordAlignment], break_after: dict[int, bool],
                     n_fillers: int, n_repetitions: int, sr: int = SAMPLE_RATE) -> tuple[dict, list[dict]]:
    """`words` is the forced alignment (in spoken order); `break_after[ref_idx]` marks punctuation."""
    dur = wav.size / sr
    mask = speech_mask(wav, sr)
    voiced = np.where(mask)[0]
    if voiced.size == 0 or not words:
        return {"duration_s": dur, "no_speech": 1.0, "initial_silence_s": dur}, []

    t0, t1 = voiced[0] * HOP_S, (voiced[-1] + 1) * HOP_S
    inner = mask[voiced[0]:voiced[-1] + 1]
    silences = [((a + voiced[0]) * HOP_S, (b + voiced[0]) * HOP_S) for a, b in _runs(inner, False)]
    phonation = inner.sum() * HOP_S
    span = max(t1 - t0, 1e-3)

    # assign each pause to the word boundary it falls in, then judge it against punctuation
    pauses = []
    for a, b in silences:
        d = b - a
        if d < 0.25:
            continue
        mid = (a + b) / 2
        prev = [w for w in words if w.end <= mid + 0.05]
        after_ref = prev[-1].ref_idx if prev else None
        ok = after_ref is not None and break_after.get(after_ref, False)
        pauses.append({"start": round(float(a), 2), "end": round(float(b), 2), "dur": round(float(d), 2),
                       "after_word": prev[-1].word if prev else None, "after_ref_idx": after_ref,
                       "at_punctuation": bool(ok)})

    n_syll = sum(is_vowel(p.phone) for w in words for p in w.phones)
    n_words = len(words)
    bad = [p for p in pauses if not p["at_punctuation"]]
    runs_between = len([p for p in pauses if p["dur"] >= 0.4]) + 1

    # rhythm: variability of word durations (normalised by phone count)
    per_phone = np.array([(w.end - w.start) / max(len(w.phones), 1) for w in words])
    feats = {
        "duration_s": dur,
        "no_speech": 0.0,
        "initial_silence_s": t0,
        "trailing_silence_s": dur - t1,
        "speech_span_s": span,
        "speech_rate_syl_s": n_syll / span,
        "articulation_rate_syl_s": n_syll / max(phonation, 1e-3),
        "words_per_s": n_words / span,
        "phonation_ratio": phonation / span,
        "n_pauses_025": len(pauses),
        "n_pauses_05": sum(p["dur"] >= 0.5 for p in pauses),
        "n_pauses_1": sum(p["dur"] >= 1.0 for p in pauses),
        "n_bad_pauses": len(bad),
        "n_bad_pauses_05": sum(p["dur"] >= 0.5 for p in bad),
        "bad_pause_time_s": float(sum(p["dur"] for p in bad)),
        "max_pause_s": max((p["dur"] for p in pauses), default=0.0),
        "mean_length_of_run_words": n_words / runs_between,
        "n_fillers": n_fillers,
        "n_repetitions": n_repetitions,
        "word_dur_cv": float(per_phone.std() / (per_phone.mean() + 1e-6)),
    }
    feats.update(_pitch_features(wav, sr))
    return feats, pauses


def _pitch_features(wav: np.ndarray, sr: int) -> dict:
    import parselmouth

    try:
        snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
        f0 = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size < 10:
            raise ValueError
        st = 12 * np.log2(f0 / np.median(f0))
        return {"f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
                "f0_std_st": float(st.std()),
                "f0_delta_std_st": float(np.diff(st).std())}
    except Exception:
        return {"f0_range_st": 0.0, "f0_std_st": 0.0, "f0_delta_std_st": 0.0}
