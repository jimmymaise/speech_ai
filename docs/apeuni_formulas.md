# APEUni: recovered Read Aloud formulas

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
| Fluency (0–90) | function of passage letters per second of reading: PCC 0.90, MAE 6.2 | no closed form |
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

Normalising by letters reaches PCC 0.90 with questions held out; each question's own best curve reaches 0.93, so
about ±5 points depend on something not in these fields (likely pauses and rhythm in the audio). Above ~19 letters
per second the score falls again.

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
- The remaining ±5 points of fluency: needs the audio (pauses, rhythm).
- Other Speaking tasks.
