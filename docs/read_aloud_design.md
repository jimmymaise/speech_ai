# Read Aloud scorer v2: design

Goal: score Read Aloud closer to the official PTE rubric than Alfa and APEUni, with clearer feedback, while still
offering their familiar score formats. Built from what the two apps' data shows works and what does not
([alfa_formulas.md](alfa_formulas.md), [apeuni_formulas.md](apeuni_formulas.md)) and from Pearson's score guides
(PTE Academic, July 2025; PTE Core, 2026).

## Design rules from the data

### Keep

| From | Choice | Reason |
|---|---|---|
| APEUni | A 0–100 score per sound and per word; good / average / bad cut-offs | Detailed feedback; wav2vec2 GOP already gives per-phone scores |
| APEUni | Word score is its own estimate, not the mean of its sounds | One weak sound should not sink a word, nor a good mean hide a wrong word |
| APEUni | Speed measured in passage letters (or phones) per second, not words | Fair across passages: fluency 50 sits at ~12 letters/s on every APEUni question |
| Both, Pearson | Content as a gate: almost nothing read → minimum score, whatever P and F say | Pearson: content 0 → no score for the response |
| Alfa | Word colours (green / yellow / red) and pause marks (`/`) on the passage | Students and teachers already read this format |
| Alfa | Explainable rules for display (counts of good / average / bad words) | Teachers can follow and trust them |
| Both | 0–90 traits, "x out of 5", CEFR level | Familiar output |

### Avoid

| Problem | Seen in | Instead |
|---|---|---|
| Fluency = speed only; pauses, hesitations, repetitions ignored (APEUni: pauses add 0.000 PCC) | Alfa, APEUni | Fluency from rate + pauses + hesitations + repetitions + rhythm, as in Pearson's rubric |
| Pronunciation mixed with fluency (APEUni: + 0.36 × F) | APEUni | Pronunciation from sounds only |
| Content counts omissions only | APEUni | Pearson rule: each replacement, omission or insertion = one error |
| Content step that costs 10% at 6 missed words of ~53 | APEUni | Proportional to the error rate |
| Word colours from a browser speech recogniser | Alfa | Our own ASR + phone scoring, pinned versions |
| Read Aloud content counted toward Reading in PTE Academic | Alfa | Academic: Speaking only (2025 guide); Core: Reading and Speaking |
| Fabricated pronunciation on open tasks (content − random offset) | Alfa | Measure pronunciation from audio on every task |

## Pipeline

```
audio ─► 16 kHz mono
  ├─ ASR (Whisper, pinned) ─► transcript ─► alignment vs passage ─► replacements / omissions / insertions ─► CONTENT
  ├─ wav2vec2 phone CTC ─► forced alignment ─► GOP per phone ─► sound score 0–100 ─► word score 0–100 ─► PRONUNCIATION
  └─ VAD + word timings + pitch ─► rate (letters/s), pauses, hesitations, repetitions, rhythm ─► FLUENCY
                                   │
                         trait models (0–90) ─► gate by content ─► Read Aloud score
                                   │
                    outputs: native · alfa_style · apeuni_style · word/sound feedback
```

Existing modules (`pte_scorer/`): `audio`, `asr`, `content`, `phonemes`, `fluency`, `words`, `model`, `scorer`.
v2 changes the trait models and the score combination; the feature extraction stays.

## Components

### 1. Sound and word scores

- Sound score = calibrated GOP (wav2vec2-lv-60-espeak), 0–100. Cut-offs start at APEUni's (good ≥ 80, average
  70–79, bad < 70) and are tuned on teacher-marked sounds.
- Word score = a small model on its sounds, duration and ASR agreement, 0–100; good ≥ 71, average 63–70, bad ≤ 62
  as a start.
- Word colour: green / yellow / red from the word score; red also for omitted or replaced words.
- Sound feedback keeps the current fields: position (start / middle / ending), `replaced` (+ heard sound) or
  `missing`; weak forms of function words are not errors.

### 2. Content (0–90 and 0–5)

- Errors = replacements + omissions + insertions (hesitations, filled pauses, leading or trailing words ignored, as
  Pearson states for Repeat Sentence).
- Content = 90 × max(0, 1 − errors / passage words), reported also on 0–5.
- Gate: content below a floor (start: < 50% of the passage read) → Read Aloud score 10, traits still shown for
  feedback.

### 3. Fluency (0–90)

Features: letters per second of speaking span, articulation rate, number and length of pauses inside phrases
(pauses at punctuation weigh less), filled pauses ("uh", "um"), repetitions and false starts from the alignment,
leading silence, rhythm (variation of syllable durations), phrasing. Model: gradient boosting trained on expert
fluency ratings (speechocean762 now, teacher ratings later), calibrated to 0–90.

Target behaviour: same total time, more mid-phrase pauses → lower fluency (APEUni and Alfa give the same score).

### 4. Pronunciation (0–90)

From sound and word scores only: mean and low percentiles of sound scores, share of bad sounds, ending-consonant
errors, stress. No fluency input. Trained on expert pronunciation ratings, calibrated to 0–90.

### 5. Read Aloud and Speaking score

- Native: Read Aloud score = gate(content) × weighted P and F. Weights are not published by Pearson; start with
  equal weights and fit them on real PTE score reports when available.
- Speaking estimate over a full mock test: task weights from Pearson's partner document (Describe Image 31%,
  Summarize Group Discussion 19%, Repeat Sentence 16%, Retell Lecture 13%, Respond to a Situation 13%,
  Read Aloud 9%), labelled as an estimate.

### 6. Output modes

| Mode | Content | Fluency | Pronunciation | Score |
|---|---|---|---|---|
| `native` (default) | PTE error rule | rate + pauses + hesitations + rhythm | sounds only | gate × weighted P, F |
| `alfa_style` | 90 × green share | min(content, 35.8 × green / s) | 90 × (G + ½A − ½R) / n | 0.4 F + 0.6 P (Academic), 0.6 F + 0.4 P (Core) |
| `apeuni_style` | 0–5 by missed words | letters/s curve | 2.34 × sound mean + 0.36 F − 141.6 | (P + F) / 2 × content factor |

All modes share the same word and sound feedback. Compatibility modes use the recovered formulas on our sound and
word scores, so they look familiar but are not identical to the apps.

## Data

| Use | Data | Size |
|---|---|---|
| Pre-train word / sound labels | Alfa word colours (CORRECT / AVERAGE / WRONG) | ~210k answers |
| Calibrate sound and word scale | APEUni per-sound and per-word scores + audio | ~4k answers, 1 with audio |
| Train trait models | speechocean762 expert ratings | 5k sentences |
| Validate and re-weight | Teacher ratings on the PTE rubric, 2 raters | 200–300 answers |
| Fit Speaking weights | Real PTE score reports with practice answers | as available |

## Validation

Benchmark: 200–300 answers (mixed levels, Academic and Core, several accents), rated by 2 teachers on the PTE
rubric. Score them with Alfa, APEUni and v2.

| Measure | Target for v2 |
|---|---|
| Correlation with mean teacher score, per trait | higher than Alfa and APEUni; ≥ 0.80 |
| Agreement with teachers vs teacher–teacher agreement | within 0.05 |
| Pause sensitivity test (same time, added mid-phrase pauses) | fluency drops |
| Word colour agreement with teachers | ≥ 85% |
| Correlation with Alfa / APEUni (sanity) | ≥ 0.80 |

## Steps

1. Run the current scorer on APEUni question 1519 (500 answers with audio); compare sound and word scores.
2. Teacher rating page; collect 200 answers × 2 raters.
3. Content rule and gate; fluency and pronunciation models retrained without speed-only shortcuts.
4. `native`, `alfa_style`, `apeuni_style` outputs.
5. Benchmark report: v2 vs Alfa vs APEUni against teachers.
