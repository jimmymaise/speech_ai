# APEUni: recovered Speaking formulas (all seven tasks)

Separate from Alfa ([alfa_formulas.md](alfa_formulas.md)). Data: 3,998 decoded APEUni answers to 8 Read Aloud
questions (ids 1506–1519, 51–64 words, ~500 answers each), all scored by engine tag `RA S-ASIA V6i`.
Fits are checked with whole questions held out. Run `python scripts/apeuni_formulas.py <read_aloud folder>`.

## Summary

| Part | Rule | Match |
|---|---|---|
| Sound score | `floor(100 × origin_score)`; good ≥ 80, average 70–79, bad < 70 | 100% of 967,916 sounds |
| Word score | `floor(83 × ai_score)`; good ≥ 71, average 63–70, bad ≤ 62 | 100% of 209,113 words |
| Read Aloud score, content 5/5 | (Pronunciation + Fluency) / 2 | 99.9% of 3,881 (rest off by 1) |
| Read Aloud score, content 4/5 | (P + F) / 2 × 0.9, rounded | 5 / 5 |
| Read Aloud score, content 2/5 | (P + F) / 2 × 0.55, rounded | 3 / 3 |
| Read Aloud score, content 0/5 | 10 | 109 / 109 |
| Content (0–5) | missed words: 0–5 → 5; 6–7 → 4; 8–9 → 2; ≥ 11 → 0 (10 and levels 1, 3 not seen) | all answers |
| Fluency (0–90) | function of passage letters per second of reading (+ a little sound quality); pauses do not count: PCC 0.90–0.93, MAE 5.3–6.2 | no closed form |
| Pronunciation (0–90) | ≈ 2.34 × mean sound score + 0.36 × F − 141.6: PCC 0.96, MAE 3.6 | no closed form |

## Fields

| Field | Meaning |
|---|---|
| `scores.components` | content (0–5), pronunciation (0–90), fluency (0–90) |
| `total_score` | Read Aloud score shown (10–90) |
| `extra_details.pronunciation_score` | raw pronunciation (0–100); not monotonic with the shown score, PCC 0.75 |
| `details[]` | per passage word: `content_word` (`ai_score`, `score`, `score_level`, `beg_pos`/`end_pos`) or `omission` |
| `details[].syllables[]` | per sound: IPA symbol, `origin_score` (0–1), `score`, `score_level` |
| `questions.json` `prediction_text` | the passage |

Word timings are in 10 ms units and run back to back from 0: silence is inside the words, there are no pause marks.
The word score is not built from its sounds (PCC 0.31 with the mean sound score, 0.9996 with `ai_score`).

## Content

Content is a gate, not a score that adds points, and it is harsh: 6 missed words out of ~53 already costs 10%.
Counts and shares fit equally well on these lengths (≤ 5 missed ≈ ≥ 90% read → 5; 8–9 missed ≈ 85% → 2).
Passages of very different lengths are needed to tell the two apart.

## Fluency

Within one question, reading time alone gives PCC 0.86–0.95. Across questions the curve shifts with the passage:
the time for fluency 50 ranges from 22.5 s to 30.4 s for passages of 51–64 words, but in letters per second it is
always ~12 (11.5–12.3). So fluency is speed measured in passage letters (or sounds) per second, not words.

| Letters per second | ≤ 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17–18 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean fluency | 10–14 | 22 | 32 | 45 | 54 | 62 | 68 | 73 | 76 | 77–79 |

Normalising by letters reaches PCC 0.90 with questions held out; each question's own best curve reaches 0.93.
Above ~19 letters per second the score falls again.

Pauses do not count. On the audio of question 1519 (469 full answers, energy-based voice detection), pause counts,
total pause time, the longest pause and leading/trailing silence add nothing once reading time is known
(PCC 0.951 → 0.950). Pauses lower fluency only by making the reading longer. Part of the rest comes from sound
quality: adding the mean sound score gives PCC 0.964 (MAE 4.0) on 1519 and 0.93 (MAE 5.3) with questions held out.
The remaining ±4 points are not explained by any field or audio measure tried.

## Pronunciation

| Inputs (questions held out) | PCC | MAE |
|---|---|---|
| `extra_details.pronunciation_score` | 0.75 | 8.2 |
| Mean sound score | 0.86 | 6.1 |
| Mean sound score + Fluency (linear) | 0.96 | 3.5 |
| All fields (gradient boosting) | 0.97 | 3.0 |

The linear weights are the same on one question and on eight (2.2–2.3 per sound point, 0.34–0.36 per fluency
point), so pronunciation is a fixed blend of sound quality and fluency. Reading slowly lowers both scores.

## Comparison

| | APEUni | Alfa | Official PTE |
|---|---|---|---|
| Unit scored | every sound and word, 0–100 | word colour from a Google transcript | sound level (Versant) |
| Content | 0–5, gate only | 90 × green share, counts toward Reading | each replaced / missed / added word = 1 error |
| Fluency | letters per second | green words per second | rhythm, phrasing, pauses, hesitations |
| Pronunciation | sound scores + fluency | 90 × (G + ½A − ½R) / n | sound level, separate from fluency |
| Read Aloud score | (P + F) / 2 | 0.4 F + 0.6 P (Academic) | not published |

Content not counting fits Pearson's 2025 rule: Read Aloud counts only toward Speaking in PTE Academic.

## Still unknown

- Other engine tags (all data is `S-ASIA V6i`).
- Content at 10 missed words and levels 1 and 3; count or share (needs much shorter and longer passages).
- The remaining ±4 points of fluency (not pauses: checked on audio).
- Other Speaking tasks (Repeat Sentence below).

# Repeat Sentence

Data: 2,500 answers to 5 sentences (ids 3404–3411, 7–13 words, prompt audio 2.8–4.9 s), engine `RS S-ASIA V3`,
plus the audio of all 2,500 answers and the 5 prompts. The format differs from Read Aloud: content is 0–3 in steps
of 0.1, there is a `recognized_text`, words are `content_word` (score 0–100 with decimals), `omission` or
`incorrect_order`, and there are no sound scores or word timings.

| Part | Rule | Match |
|---|---|---|
| Content (0–3) | 3 − the sum of per-word penalties for missed or out-of-order words; function words cost much less than content words | MAE 0.00–0.02 with ≤ 3 words missed (per-sentence weights) |
| Repeat Sentence score, content 3/3 | ≈ 47 + 0.242 × (P + F) | 76% exact, 100% within 1 |
| Repeat Sentence score, content < 3 | base and weight on P + F both fall with content | model on C, P, F: MAE 1.1 |
| Pronunciation | word scores + fluency (word mean alone: PCC 0.33) | PCC 0.88–0.90, MAE 3.4–3.7 |
| Fluency | mostly content and word scores; speed and pauses matter little | PCC 0.82, MAE 5.5 |

## Content weights

Penalty per missed word, fitted per sentence:

| Sentence | Function word | Content word |
|---|---|---|
| The supermarket is located near the theater. | 0.10–0.12 | 0.39–0.43 |
| The school provides hot lunches for students. | 0.14 | 0.32–0.36 |
| The telephone booths are available for all employees in the office. | 0.06–0.08 | 0.24–0.26 |
| Your application for accommodation must be submitted before the course starts. | 0.10 | 0.20 |
| Lack of citation and evidence in an article will be regarded as plagiarism. | 0.07–0.11 | 0.20–0.22 |

A content word is worth 2–4 function words, varying by sentence, so the weights likely come from a word-importance
model. Penalties for all words add up to about 2, not 3, and with many words missed the score falls faster than
the sum (minimum seen 0.2). Words said out of order cost the same as missed words. This is finer than Pearson's
rule (3 = all words in sequence, 2 = ≥ 50%, 1 = < 50%, 0 = almost nothing) and worth copying.

## Fluency

Checked on all 2,500 recordings with energy-based voice detection, against the prompt length.

| Inputs (sentences held out) | PCC | MAE |
|---|---|---|
| Letters per second of speech | 0.16 | 8.8 |
| Answer length ÷ prompt length | 0.20 | 8.7 |
| + pauses, leading silence | 0.22 | 8.9 |
| Content, words repeated, mean word score | 0.81 | 5.6 |
| All | 0.82 | 5.5 |

With full content, fluency moves only from ~77 (answer shorter than the prompt) to ~69 (more than 1.5× the
prompt), and the mean differs by sentence (69–80). Unlike Read Aloud, Repeat Sentence fluency is not a speed score;
it follows how much was repeated and how well, and the rest (±5) is not explained by timing or pauses. It is likely
a learned model.

## Read Aloud vs Repeat Sentence

| | Read Aloud | Repeat Sentence |
|---|---|---|
| Content | missed-word count, gate only | weighted words, part of the score |
| Score | (P + F) / 2 | ≈ 47 + 0.24 (P + F) at full content |
| Pronunciation | sound scores + fluency | word scores + fluency |
| Fluency | letters per second; pauses do not count | mostly content and word scores; speed and pauses matter little |

## Repeat Sentence: status and data needed

| Part | Status |
|---|---|
| Content (0–3) | Rule known (3 − per-word penalties); weights known for the 5 sentences, not how APEUni sets them for a new sentence |
| Score, content 3/3 | Formula: ≈ 47 + 0.242 × (P + F), always within 1 |
| Score, content < 3 | Model only (MAE 1.1) |
| Pronunciation, fluency | Likely APEUni's own trained models; approximations only (±3.5, ±5.5) |

Data that would improve it (answers JSON + `questions.json` only; no answer audio needed):

1. 15–20 more sentences, 200–300 answers each, mixed lengths (5–15 words) and vocabulary. Enough words to learn how
   the per-word weight is set (function vs content word, word frequency, length, position).
2. Sentences where many students miss words (long or hard ones), to pin down how content falls when most words are
   missed and the score formula for content < 3.
3. Any answers with an engine tag other than `RS S-ASIA V3`.

# Retell Lecture

Data: 3,900 answers to 13 lectures (ids 688–724, 300 each), engine `RL S-ASIA V3.1`. Traits are on PTE scales:
content 0–6, pronunciation 0–5, fluency 0–5, all in steps of 0.1. `prediction_text` is the lecture transcript,
`recognized_text` the student's transcript. Words are what the student said (no reference passage) with timings and
sound scores, scored like Read Aloud (`floor(83 × ai_score)`).

| Part | Rule | Match |
|---|---|---|
| Retell Lecture score, content ≥ 4 | 10 + 8 × (P + F) | 83% exact, 99.6% within 1 (all answers: 79%, 97%) |
| Content < 4 | a small cut: the 8 drops to ~7.6 at content 2 and ~6.7 near 0 | MAE 0.5 overall |
| Content (0–6) | not keywords or length; likely a semantic or LLM judge. Comment "Good content." exactly when content ≥ 5 | keyword overlap PCC 0.50; + length 0.68; + P, F 0.76 |
| Pronunciation (0–5) | mean sound score + fluency, like Read Aloud | PCC 0.90, MAE 0.35 |
| Fluency (0–5) | words per second plus long words (pauses inside word timings), length and sound quality | PCC 0.76, MAE 0.43 |

Content counts very little in the score: two answers with the same P and F differ by at most a few points whatever
their content. Pearson's guide makes content the first gate for Retell Lecture (content 0 → no score) and has a
human expert review it; APEUni does the opposite.

Fits hold out whole lectures (13).

# Describe Image, Respond to a Situation, Summarize Group Discussion, Answer Short Question

| Task | Data | Engine | Scales |
|---|---|---|---|
| Describe Image | 2,500 answers, 5 images | `DI S-ASIA V3` | content 0–6, P and F 0–90 |
| Respond to a Situation | 900 answers, 3 situations | `RTS S-ASIA V2.1` | content 0–6, P and F 0–90 |
| Summarize Group Discussion | 900 answers, 3 discussions | `SGD S-ASIA V3` | content 0–6, P and F 0–5 |
| Answer Short Question | 1,500 answers, 5 questions | none | content 0–1 |

Word and sound scores in Describe Image and Summarize Group Discussion follow the Read Aloud rules. Respond to a
Situation gives each spoken word a 0–100 score and a 0–1 `tc_fluency`, with `tc_pronun` and `tc_fluency` totals.

## Task score

| Task | Rule | Match |
|---|---|---|
| Describe Image | (P + F) / 2; content is not used at all | 99.1% exact, 99.9% within 1 |
| Respond to a Situation | (P + F) / 2 × content / 6 | 95% exact, 100% within 1 |
| Summarize Group Discussion | no clean rule; content, P and F all count | best model MAE 3.5–4.0 |
| Answer Short Question | = content (0 or 1) | 100% |

## Content

| Task | Rule | Match |
|---|---|---|
| Respond to a Situation | **word count only**: min(6, round(0.12 × words, 1)); 50+ words → 6 | 98.9% exact, 99.3% within 0.1 |
| Answer Short Question | 1 if **any single word** of the answer key is in the transcript (key split on `/` and spaces; extra words allowed; no stemming, so "forks" for "fork" fails) | 1,500 / 1,500 |
| Describe Image | not keywords (overlap with the sample answer: PCC 0.13) nor length (0.41); likely an AI judge | — |
| Summarize Group Discussion | partly keywords and length (PCC 0.69); comment "Good content." when content ≥ 5 | — |

## Pronunciation and fluency

| Task | Pronunciation | Fluency |
|---|---|---|
| Describe Image | sound mean + F: PCC 0.84 | speech rate, timing: 0.70; + pronunciation and content: 0.79 |
| Respond to a Situation | word mean + F: PCC 0.94 | `tc_fluency` and timing: 0.90 |
| Summarize Group Discussion | sound mean (+ F): PCC 0.91–0.93 | timing: 0.65; + pronunciation and content: 0.81 |

Fits hold out whole questions.

## Against Pearson's rules

- Describe Image: Pearson scores content first and gives no score when it is 0 (and a human reviews it). APEUni
  ignores content in the score.
- Respond to a Situation: Pearson's content asks whether the response handles the situation. APEUni's content is
  the word count: 50 words on anything earn full content.
- Answer Short Question: same correct / incorrect shape as Pearson, but the matching is crude. Every word of a
  multi-word key counts on its own, including "a": for "A black eye / bruise", the answers "a bruce", "it may be a
  little pinch of water" and "gotta be a bitch" all score 1. Plurals are not accepted unless listed ("photos" is
  listed, "forks" is not). Answers listing several options ("spoon and fork") also score 1.
