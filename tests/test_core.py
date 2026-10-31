import numpy as np
import pandas as pd

from pte_scorer.content import align_words, content_features
from pte_scorer.dataset import validate
from pte_scorer.phonemes import ctc_forced_align, phone_error_rate
from pte_scorer.scorer import rule_scores
from pte_scorer.text import tokenize_hypothesis, tokenize_reference


def test_reference_tokens_and_punctuation():
    ref = tokenize_reference("In 1990, sea-level rise was 25% higher.")
    assert [r.text for r in ref] == ["in", "nineteen", "ninety", "sea", "level", "rise", "was",
                                     "twenty", "five", "percent", "higher"]
    breaks = [r.text for r in ref if r.break_after]
    assert breaks == ["ninety", "higher"]


def test_hypothesis_tokens():
    assert tokenize_hypothesis("Um, the cat's  mat.") == ["um", "the", "cat's", "mat"]


def test_alignment_ops():
    ops = align_words("the quick brown fox".split(), "the quik fox jumps".split())
    assert [o.op for o in ops] == ["match", "sub", "del", "match", "ins"]


def test_content_features_fillers_and_repetition():
    ref = "rising temperatures threaten ecosystems".split()
    hyp = "rising um temperatures temperatures threaten".split()
    f, _ = content_features(ref, hyp)
    assert f["n_fillers"] == 1
    assert f["n_repetitions"] == 1
    assert f["pct_omit"] == 0.25
    assert f["pct_passage_covered"] == 0.75


def test_ctc_forced_align_toy():
    # vocab: 0=blank, 1=a, 2=b ; frames clearly say: a a _ b b
    T, V = 5, 3
    lp = np.full((T, V), np.log(0.05))
    for t, k in enumerate([1, 1, 0, 2, 2]):
        lp[t, k] = np.log(0.9)
    path = ctc_forced_align(lp, [1, 2], blank=0)
    assert list(path) == [0, 0, -1, 1, 1]
    # repeated token needs a blank in between: 'a a' cannot fit in 1 frame
    assert ctc_forced_align(lp[:1], [1, 1], blank=0) is None


def test_phone_error_rate():
    assert phone_error_rate(list("abc"), list("abc")) == 0
    assert phone_error_rate(list("abcd"), list("abxd")) == 0.25


def test_rule_scores_monotonic():
    good = {"n_ref_words": 50, "n_errors": 0, "n_bad_pauses": 0, "speech_rate_syl_s": 4.2,
            "pct_phones_bad_25": 0.02, "phone_error_rate": 0.1, "f0_std_st": 3}
    bad = {**good, "n_errors": 12, "n_bad_pauses": 4, "speech_rate_syl_s": 2.2,
           "pct_phones_bad_25": 0.3, "phone_error_rate": 0.5}
    g, b = rule_scores(good), rule_scores(bad)
    assert all(g[t] > b[t] for t in g)
    assert g["content"] == 5
    assert rule_scores({"no_speech": 1}) == {"content": 0, "fluency": 0, "pronunciation": 0}


def test_validate_flags_problems(tmp_path):
    pd.DataFrame({"passage_id": ["p1"], "reference_text": ["one two three four five six"]}).to_csv(
        tmp_path / "passages.csv", index=False)
    pd.DataFrame({"response_id": ["r1", "r1"], "passage_id": ["p1", "p9"], "speaker_id": ["s1", "s2"],
                  "audio_path": ["a.wav", "b.wav"], "content": [5, 7], "fluency": [4, 4],
                  "pronunciation": [4, None]}).to_csv(tmp_path / "labels.csv", index=False)
    issues = " | ".join(validate(tmp_path).issue)
    for expected in ("duplicate response_id", "unknown passage_id", "outside 0-5", "missing pronunciation",
                     "audio not found"):
        assert expected in issues


def test_levels():
    from pte_scorer.levels import cefr, estimate, to_pte_scale
    assert to_pte_scale(0, 5.0) == 10 and to_pte_scale(5, 5.0) == 90
    assert to_pte_scale(0, 90.0) == 10          # Alfa's 0 (no answer) -> PTE minimum 10
    assert to_pte_scale(64, 90.0) == 64
    assert [cefr(x) for x in (10, 29, 30, 43, 58, 59, 75, 76, 85, 90)] == \
        ["A1", "A1", "A2", "B1", "B1", "B2", "B2", "C1", "C2", "C2"]
    e = estimate({"content": 70, "fluency": 60, "pronunciation": 62})
    assert e["read_aloud_10_90"] == 64 and e["cefr"] == "B2" and "not your PTE score" in e["note"]


def test_word_report_and_alfa_tags():
    from pte_scorer.words import DEFAULTS, marked_pauses, tagged_answer, word_report
    cfg = dict(DEFAULTS)
    details = {
        "ref_words": ["A", "cat", "asked,", "them."],
        "ops": [{"op": "match"}, {"op": "match"}, {"op": "sub", "said": "ask"}, {"op": "del"}],
        "words": [
            # "a" said as /ə/: normal weak form, not an error
            {"word": "a", "ref_idx": 0, "start": 0, "end": .1, "phones": [["eɪ", -6.0, "ə", 0.0, 0.9]]},
            # final /t/ never heard and nothing else there -> missing ending sound
            {"word": "cat", "ref_idx": 1, "start": .1, "end": .4,
             "phones": [["k", 0, "g", 1, 0], ["æ", 0, "ɛ", 1, 0], ["t", -7.0, "d", 0.0, 0.01]]},
            {"word": "ask", "ref_idx": 2, "start": .5, "end": .8, "phones": [["æ", 0, "ɛ", 1, 0]]},
        ],
        "pauses": [{"dur": 1.1, "after_ref_idx": 1, "at_punctuation": False},
                   {"dur": 0.8, "after_ref_idx": 2, "at_punctuation": True}],
    }
    w = word_report(details, cfg)
    assert [x["label"] for x in w] == ["CORRECT", "WRONG", "WRONG", "WRONG"]
    assert w[1]["issues"][0]["problem"] == "missing" and w[1]["issues"][0]["position"] == "ending"
    assert w[2]["status"] == "substituted" and w[3]["status"] == "omitted"
    p = marked_pauses(details, cfg)
    assert p == {1: 2}
    assert tagged_answer(w, p) == ("[CORRECT]a[/CORRECT] [[WRONG]]cat[[/WRONG]] [[PAUSE]]//[[/PAUSE]] "
                                   "[[WRONG]]asked them[[/WRONG]]")
