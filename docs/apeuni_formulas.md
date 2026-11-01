# APEUni: recovered Read Aloud formulas

Separate from Alfa ([alfa_formulas.md](alfa_formulas.md)). Data: 500 decoded APEUni answers to one Read Aloud
question (model_id 1519, 53 words), all scored by engine tag `RA S-ASIA V6i`. Check with
`python scripts/apeuni_formulas.py <answers.json>`.

## Summary

| Part | Rule | Match |
|---|---|---|
| Sound score | `floor(100 × origin_score)`; good ≥ 80, average 70–79, bad < 70 | 100% |
| Word score | `floor(83 × ai_score)`; good ≥ 71, average 63–70, bad ≤ 62 | 100% of 25,185 words |
| Read Aloud score, content 5/5 | (Pronunciation + Fluency) / 2 | 469 / 469 |
| Read Aloud score, content 4/5 | (P + F) / 2 × 0.9, rounded | 3 / 3 |
| Read Aloud score, content 0/5 | 10 | 28 / 28 |
| Content (0–5) | 0 missed words → 5; 6 missed → 4; ≥ 22 missed → 0 (7–21 not seen) | only these levels seen |
| Fluency (0–90) | reading time: PCC 0.95, MAE 4.6 (all timing + word features: 0.97) | no closed form |
| Pronunciation (0–90) | ≈ 2.21 × mean sound score + 0.34 × F − 132.8: PCC 0.92, MAE 3.7 (all features + F: 0.96, 2.7) | no closed form |

## Fields

| Field | Meaning |
|---|---|
| `scores.components` | content (0–5), pronunciation (0–90), fluency (0–90) |
| `total_score` | Read Aloud score shown (10–90) |
| `extra_details.pronunciation_score` | raw pronunciation (0–100); not monotonic with the shown score (76.7 → 25, 71.9 → 60), PCC 0.69 |
| `details[]` | per passage word: `content_word` (`ai_score`, `score`, `score_level`, `beg_pos`/`end_pos`) or `omission` |
| `details[].syllables[]` | per sound: IPA symbol, `origin_score` (0–1), `score`, `score_level` |

Word timings are in 10 ms units and run back to back from 0: silence is inside the words, there are no pause marks.
The word score is not built from its sounds (PCC 0.31 with the mean sound score, 0.9996 with `ai_score`).

## Fluency by reading time (53 words, content 5/5)

| Seconds | ≤ 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 33 | ≥ 37 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean fluency | ~80 | 75 | 70 | 61 | 52 | 44 | 34 | 25 | 18 | 10–13 |

About −4.5 points per extra second. ~3.3 words/s earns 80–90.

## Comparison

| | APEUni | Alfa | Official PTE |
|---|---|---|---|
| Unit scored | every sound and word, 0–100 | word colour from a Google transcript | sound level (Versant) |
| Content | 0–5, gate only | 90 × green share, counts toward Reading | each replaced / missed / added word = 1 error |
| Fluency | reading time | green words per second | rhythm, phrasing, pauses, hesitations |
| Pronunciation | sound scores + fluency | 90 × (G + ½A − ½R) / n | sound level, separate from fluency |
| Read Aloud score | (P + F) / 2 | 0.4 F + 0.6 P (Academic) | not published |

Content not counting fits Pearson's 2025 rule: Read Aloud counts only toward Speaking in PTE Academic.

## Still unknown

- Other passages and lengths (all data is one question); other engine tags and their calibration.
- Content cut-offs between 7 and 21 missed words (levels 1–3 never seen); the 0.9 factor rests on 3 answers.
- The rest of fluency (±4.6 points) and pronunciation; other Speaking tasks.

Next data: 500 answers from each of 3–5 more Read Aloud questions of different lengths, ideally other engine tags.
