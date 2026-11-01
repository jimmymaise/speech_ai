# How the source scores each Speaking task

Recovered from the source's own outputs in the crawl of 31 Oct 2026 (about 780k scored answers).
Check the Read Aloud / Repeat Sentence formulas with `scripts/alfa_formulas.py`; the open-task numbers come from
`scripts/collect_open_alfa.py` plus the analysis described below.

Notation: per answer, **G / A / R** = number of passage words the source tagged CORRECT (green) / AVERAGE (yellow) /
WRONG (red), **n** = G + A + R. "round" is round half up. All traits are on 0–90 unless noted.

> Crawl folders `respond_to_situation/` and `summarize_group_discussion/` are swapped: the first holds Summarize
> Group Discussion (a group-discussion transcript, ~200-word answers), the second Respond to a Situation
> (a situation prompt, ~85-word answers). Names below are the real task names.

## Summary

| Task | Content | Pronunciation | Fluency | Speaking score |
|---|---|---|---|---|
| Read Aloud | **exact**: round(90·G/n) | **exact**: 90·(G + ½A − ½R)/n | approx.: content capped by speed | **exact**: 0.4·F + 0.6·P (Core 0.6·F + 0.4·P) |
| Repeat Sentence | **exact**: lookup on (G, n) | **exact**: same as Read Aloud | approx.: green share × speed vs prompt | **exact**: same as Read Aloud |
| Answer Short Question | **exact**: 1 if answer correct, else 0 | = content (5 or 0 out of 5) | = content | none |
| Retell Lecture | 7 levels from keyword matches (~75%) | content − random 1–13 | **≈** min(90, 30·words/s) | **exact (84%, 99.5% ±1)**: 0.6·S + 0.4·F, S = hidden value per level |
| Summarize Group Discussion | 7 levels from keyword matches (~79%) | content − random 1–13 | **≈** min(90, 34·words/s) | **exact (90%)**: 0.6·C + 0.4·F |
| Describe Image | related to keyword matches, no fixed rule | content − random 0–18 | **≈** min(90, 30·words/s) | 0.6·C + 0.4·F (Core 0.4·C + 0.6·F) |
| Respond to a Situation | no keyword marks, no rule found (likely an AI judge) | content − random 0–19 | **≈** min(90, 30·words/s) | **exact (83–90%)**: 0.6·C + 0.4·F |

"exact" = the formula reproduces the source's number exactly for ≥ 97% of answers.

## Word colours (Read Aloud, Repeat Sentence)

Each passage word is compared with the source's transcript (`google_transcript`):

| Colour | Rule | Agreement on 123k Repeat Sentence words |
|---|---|---|
| green | the word is in the transcript | 98% of green words |
| red | the word is missing, or replaced by a clearly different word | 95% of red words |
| yellow (1) | replaced by a similar word (string similarity ≥ 0.7, e.g. *activity* for *activities*) | |
| yellow (2) | the word is in the transcript but still tagged yellow; 64% of yellow words. More frequent on the first word of a sentence (27% vs 8%). Not explained by the transcript; Google Cloud word confidence explains it only weakly (yellow 20% at confidence < 0.5, 6% at > 0.98) | |

These three rules alone match 88% of words (green 98%, red 95%, yellow 24%).

Score impact of a wrong colour: green ↔ yellow moves pronunciation by 0.5·90/n per word (≈ 0.8 points on a
55-word Read Aloud, 4.5 points on a 10-word Repeat Sentence); green ↔ red moves it by 1.5·90/n.

### Where the transcript comes from

Not Google Cloud Speech-to-Text. On 200 held-out answers (100 Read Aloud, 100 Repeat Sentence), word agreement with
the source's transcript after normalising case, apostrophes, hyphens and fillers:

| Recogniser | Read Aloud | Repeat Sentence |
|---|---|---|
| Whisper large-v3-turbo (ours) | **0.84** | **0.90** |
| Google Cloud v1 default, en-US | 0.75 | 0.87 |
| Google Cloud v1 default, en-AU | 0.73 | 0.84 |
| Google Cloud v1 latest_long | 0.75 | 0.84 |
| Google Cloud v1 latest_short | 0.67 | 0.82 |

The error style is Google's (*crucial* → *the cruel*, *autumn* → *antonym*), profanity is masked as `*******`,
some answers lose their first words, and a separate mobile transcript field exists. Most likely the browser's
(Chrome Web Speech API) or the phone's built-in recogniser, run on the client while recording. Not confirmed.

## Read Aloud

- **Content** = round(90·G/n). 97.8% exact over 206,657 answers; the rest differ mostly by word-count differences
  (hyphenated words).
- **Pronunciation** = clip(round(90·(G + 0.5·A − 0.5·R)/n), 0, 90). 97.9% exact. Equivalently
  135·G/n + 90·A/n − 45: a red word costs, not just fails to earn.
- **"x out of 5"** = bands of pronunciation: ≥88 → 5, ≥84 → 4.5, ≥76 → 4, ≥65 → 3.5, ≥55 → 3, ≥35 → 2.5, ≥7 → 2.
- **Fluency** ≈ min(content, 35.8·G / recording seconds): content, reduced when slower than about 2.5 passage words
  per second. PCC 0.987, MAE 2.1, 70% within ±2. The remaining error is most likely a different duration measure
  (e.g. speech time from word timestamps rather than file length).
- **Speaking** = 0.405·F + 0.584·P − 0.19 (Academic), 0.600·F + 0.396·P − 0.12 (Core). Mean error < 0.4.

## Repeat Sentence

- **Content**: a fixed lookup on (G, n), not a straight line. For a 10-word sentence: G = 0…10 →
  0, 12, 23, 32, 47, 59, 62, 70, 76, 82, 90. 99.9% exact over 140,401 answers. Full table:
  `content_table()` in `scripts/alfa_formulas.py`.
- **Pronunciation**: same formula as Read Aloud. 99.6% exact.
- **Fluency**: depends on the green share G/n and on the answer length relative to the prompt audio length
  (`data/alfa_rs/prompt_durations.csv`). Answers with every word green get 90 while the recording is ≤ ~1.3× the
  prompt, then ~86 at 1.5–1.75×, ~76 at 1.75–2×, ~66 at 2–2.5×, ~39 at 3–4×. Even when fast, only answers with
  ≥ 80% green words reliably get 90. No closed form; a small tree on (G, n, recording s, prompt s) gives PCC 0.992,
  MAE 1.6.
- **Speaking**: as Read Aloud (Academic 0.419·F + 0.586·P, Core 0.600·F + 0.399·P).
- **Listening** ≈ content (1.00·C − 0.4).

## Answer Short Question

content = fluency = 1 and pronunciation "5 out of 5" exactly when the expected answer is tagged `[CORRECT]`;
otherwise 0 / 0 / "0 out of 5". 100% of 111,925 answers.

## Open-answer tasks

### Pronunciation is not measured

pronunciation = content − an offset that correlates with nothing measured (fluency, length, pauses, keyword
matches: |r| ≤ 0.05 for Retell Lecture and Describe Image). Retell Lecture and Summarize Group Discussion offsets are
roughly uniform in −5…−13; at content 90 the offset is −1…−8. To match the source: pronunciation ≈ content − 8.
Our wav2vec2 pronunciation measure is a real measurement and will not match these numbers.

### Content

- **Retell Lecture, Summarize Group Discussion**: only 7 values, 18 / 27 / 45 / 59 / 68 / 77 / 90 (0 for no
  answer), a step function of the number of distinct matched keywords, with thresholds per question. From our
  rebuilt keyword lists (the union of words the source marked `[[MATCH]]` across all answers to a question) the
  level is 75% (Retell Lecture) and 79% (Summarize Group Discussion) exact; the misses sit at level boundaries.
- **Describe Image**: correlated with keyword matches (r 0.70) but no fixed rule (46% exact per question and match
  count). Values cluster at 90 / 85 / 79 / 74 / 68.
- **Respond to a Situation**: no keyword marks. Transcript counts explain little (PCC 0.31). Answers carry an
  `aiFeedbackComment`, so content is likely an AI judge. Plan: score with an LLM rubric.

### Speaking and listening

- Speaking = 0.6·C + 0.4·F for Summarize Group Discussion (90% exact), Respond to a Situation (83–90%) and
  Describe Image Academic (Core: 0.4·C + 0.6·F).
- Retell Lecture speaking = 0.6·S + 0.4·F where S is a hidden value per content level
  (45 → 33, 59 → 52, 68 → 64, 77 → 77, 90 → 90): 84% exact, 99.5% within ±1.
- Listening ≈ content. Retell Lecture levels map to 15 / 30 / 45 / 60 / 68 / 75 / 90.

### Fluency

Speaking speed only: fluency ≈ min(90, k · transcript words / recording seconds). Pauses, pause marks and pause
length add nothing once speed is known. Measured on ~1,700 answers per task (words from the source's transcript,
seconds from the mp3):

| Task | k | full 90 at | PCC | MAE | within ±2 |
|---|---|---|---|---|---|
| Describe Image | 30.4 | ~3.0 words/s (180 wpm) | 0.993 | 0.81 | 95% |
| Retell Lecture | 30.3 | ~3.0 words/s | 0.993 | 0.84 | 93% |
| Respond to a Situation | 30.3 | ~3.0 words/s | 0.988 | 0.88 | 94% |
| Summarize Group Discussion | 34.0 | ~2.65 words/s | 0.990 | 0.86 | 93% |

So k·words/s ≈ wpm/2 (wpm/1.76 for Summarize Group Discussion). Speaking faster always helps, whatever is said.

## Still open

| Item | Next step |
|---|---|
| Exact fluency for Read Aloud / Repeat Sentence | try speech-time measures instead of file length |
| Yellow words (case 2) | needs the source recogniser's word confidence; not available |
| Retell Lecture / Summarize Group Discussion thresholds | with our rebuilt keyword lists the best is 74–78% exact, < 2% off by more than one level; the source's real lists are not in the crawl |
| Describe Image content | test keyword weights and image-specific lists; otherwise a model or LLM |
