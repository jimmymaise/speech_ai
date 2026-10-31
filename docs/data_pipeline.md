# Data pipeline: from crawled answers to trained models

All steps are resumable. Re-run the same command after an interruption and it continues where it stopped.

```
crawl (<task>/<question_id>/question.json + sample_response_{ac,core}.json)
  │  scripts/prepare_alfa.py --task <task>        → labels.csv, passages.csv, word_labels.jsonl
  ▼
pool (all scored answers of a task)
  │  scripts/sample_alfa.py                       → selected.csv (balanced, de-duplicated, split)
  ▼
round (selected answers)
  │  scripts/prepare_alfa.py --download --selected → audio/
  │  python -m pte_scorer extract                 → one JSON per answer in the cache + features parquet
  ▼
features
  │  python -m pte_scorer train                   → score models (LightGBM per trait)
  │  scripts/train_word_labels.py                 → word classifier (CORRECT / AVERAGE / WRONG)
  │  scripts/eval_holdout.py                      → held-out report against the targets
```

## 1. Convert the crawl

```bash
python scripts/prepare_alfa.py $CRAWL_DIR data/alfa --task read_aloud
python scripts/prepare_alfa.py $CRAWL_DIR data/alfa_rs --task repeat_sentence
```

Tasks: `read_aloud`, `repeat_sentence`, `answer_short_question`, `describe_image`, `retell_lecture`,
`respond_to_situation`, `summarize_group_discussion`.

Output per task:

| File | Content |
|---|---|
| `labels.csv` | one row per answer: ids, `audio_url`, `content` / `fluency` / `pronunciation` (0–90), `pron_5`, `speaking`, `listening`, `reading`, `pte_type` (1 Academic, 2 Core), engine version, flags |
| `passages.csv` | one row per question: `reference_text` (passage, sentence, lecture transcript, expected answer), prompt audio / image |
| `word_labels.jsonl` | the source's word labels (`CORRECT` / `AVERAGE` / `WRONG` / `PAUSE`) and, for open-answer tasks, the keywords it marked as matched |

Personal data: student names and images are dropped; student ids are re-hashed (same hash across tasks).

Response files that only say `{"same_as": ...}` are skipped: the other file of that question holds the same answers.

## 2. Pick a balanced sample

```bash
python scripts/sample_alfa.py data/alfa --per-band 6 --max-per-student 15 --core-share 0.5
```

Rules:

1. Drop answers without scores and `is_inflated == 1`.
2. One **random** attempt per (student, question). A student's best attempt would bias scores upward.
3. Level bands on the mean of the three traits (0–9, 10–29, 30–42, 43–50, 51–58, 59–65, 66–75, 76–84, 85–90).
   Per question take up to `--per-band` answers from each band, so weak and strong speakers are both covered.
4. A student contributes at most `--max-per-student` answers.
5. About `--core-share` of each band goes to PTE Core answers when available; unused slots are back-filled with Academic.
6. Split by student hash: train 80% / val 10% / test 10%. About 10% of questions are held out as `test_passage`
   and only answered by val/test students, so it shares neither students nor questions with train.

Then make a round folder whose `labels.csv` is the selection with `audio_path` pointing into the pool:

```python
import pandas as pd, shutil
s = pd.read_csv("data/alfa/selected.csv")
s["audio_path"] = "../alfa/" + s.audio_path
s.to_csv("data/alfa_r1/labels.csv", index=False)
shutil.copy("data/alfa/passages.csv", "data/alfa_r1/")
```

## 3. Download and extract

```bash
CRAWL_DIR=/path/to/out_answers scripts/run_round.sh read_aloud data/alfa data/alfa_r1 cache/alfa_r1
```

About 235 KB of audio per Read Aloud answer. Extraction takes about 1.7–2.3 s per Read Aloud answer on an M4 Pro.
The cache keeps the full analysis per answer (features, alignment, per-phone scores), so word-level tuning can be
redone without re-extracting.

## 4. Train and evaluate

```bash
python -m pte_scorer train cache/alfa_r1_features.parquet --out models/read_aloud.joblib
python scripts/eval_holdout.py cache/alfa_r1_features.parquet --curve --out reports/read_aloud.json
python scripts/train_word_labels.py data/alfa_r1 cache/alfa_r1
```

- Model inputs are an allow-list: the audio features listed in `<parquet>.features.json` plus `pte_type`.
  Other columns from the label source (`pron_5`, `speaking`, `pause_count`, ...) are never inputs.
- One model serves Academic and Core, with `pte_type` as a feature. Metrics are reported per type.
- `eval_holdout.py` trains on `train` and reports on `test` (new students) and `test_passage`
  (new students and new questions): correlation, mean error in points, share within ±10 points, share with the
  same CEFR level, and the same on level 43+. `--curve` adds a learning curve (25 / 50 / 100% of train).

## Label cleaning

About 30% of the source's 0/0/0 Read Aloud scores are complete, correct readings (in many of them our ASR heard
over 95% of the passage). They look like scoring failures. `model.drop_label_failures` removes answers with all
traits below 10 while at least 50% of the passage was read correctly. Training, evaluation and the word classifier
all apply it.

## Facts recovered from the source's data (206k Read Aloud answers)

| Fact | Detail |
|---|---|
| Speaking score from Read Aloud | Academic ≈ 0.405·fluency + 0.583·pronunciation − 0.17; Core ≈ 0.600·fluency + 0.396·pronunciation − 0.10. Content barely counts (it goes to Reading). Mean error < 0.4 points |
| "x out of 5" pronunciation | A banding of the 0–90 pronunciation: ≥88 → 5, ≥84 → 4.5, ≥76 → 4, ≥65 → 3.5, ≥55 → 3, ≥35 → 2.5, ≥7 → 2 |
| Word colours | Mostly "was the word read" (83% agreement from that alone), plus some pronunciation |
| Pause marks | Mid-phrase pauses; the number of `/` grows with the pause length |
| Open-answer content | Number of distinct matched keywords correlates 0.70–0.80 with content (Describe Image, Retell Lecture, Respond to a Situation). Summarize Group Discussion has no keyword marks |

## Data available (crawl of 31 Oct 2026)

| Task | Questions | Scored answers with audio |
|---|---|---|
| Read Aloud | 300 | 211,644 |
| Repeat Sentence | 300 | 142,317 |
| Describe Image | 300 | 121,795 |
| Answer Short Question | 300 | 111,925 |
| Summarize Group Discussion | 300 | 86,462 |
| Retell Lecture | 300 | 64,937 |
| Respond to a Situation | 198 | 41,269 |
