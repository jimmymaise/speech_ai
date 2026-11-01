# Plan: PTE Speaking scorer

Status as of 31 Oct 2026.

## Goal

Score all seven PTE Speaking tasks close to Alfa's automatic scores, in the same output format, with clearer
word-level feedback. Then improve beyond Alfa with real PTE score reports and teacher ratings.

## Tasks

| Task | How it is scored | Status |
|---|---|---|
| Read Aloud | Word accuracy vs passage, fluency, pronunciation, word highlighting | Pilot done; full-data run in progress |
| Repeat Sentence | Same pipeline as Read Aloud | Feature extraction in progress |
| Answer Short Question | Expected answer found in the transcript | Next |
| Respond to a Situation | Meaning similarity + fluency + pronunciation (no keyword marks in the source) | Text-only content test: 0.74 |
| Retell Lecture | Keyword coverage + fluency + pronunciation | Text-only content test: 0.85 |
| Describe Image | Keyword coverage + fluency + pronunciation | Text-only content test: 0.75 |
| Summarize Group Discussion | Keyword coverage + fluency + pronunciation | Text-only content test: 0.94 correlation |

Keyword lists per question are rebuilt from the words Alfa marked as matched across all answers to that question.
New questions need a list built from the reference text or by an LLM.

## Quality targets (held-out students and questions)

| Measure | Target | Read Aloud pilot (540 answers) |
|---|---|---|
| Correlation with Alfa, per trait | ≥ 0.90 | 0.92 / 0.94 / 0.93 |
| Mean error (points out of 90) | ≤ 6 | 5.5–6.7 |
| Answers within ±10 points | ≥ 85% | 79–81% |
| Same CEFR level | ≥ 75% | to be measured |
| Word colour agreement | ≥ 80% | 85% |

## Data

No more crawling is needed (see `docs/data_pipeline.md`). We train on a balanced sample of about 13k answers
per task. Optional, to go beyond Alfa: real PTE score reports with practice answers, teacher ratings for
200–500 answers.

## Compute

Final feature extraction for all tasks: about 65k answers, 640 audio hours. On one AWS g6.4xlarge (1× L4,
16 vCPU) about 10–14 hours. Budget $40–70 including re-runs (approximate on-demand prices).

Before the GPU run: switch ASR from `mlx-whisper` (Apple only) to `faster-whisper`, so training and production
use the same recogniser.

## Serving

| Answers / month | Lambda + Whisper on CPU | GPU (server or Cloud Run) |
|---|---|---|
| 5k | ~$25 | $150–300 |
| 20k | ~$100 | $300–400 |
| 100k | ~$500 | $400–600 |
| 500k | ~$2,500 | $800–1,500 |

Scoring runs in the background: the app uploads audio, gets a job id, and receives the result a few seconds later.
Start on Lambda, move to GPU above about 100k answers a month. Same container for both.

## Timeline

| When | Step |
|---|---|
| Now | Request AWS GPU quota |
| Day 1 | GPU machine set up, faster-whisper check on 100 answers |
| Day 1–2 | Feature extraction, all tasks |
| Day 2–3 | Train + report: Read Aloud, Repeat Sentence, Answer Short Question |
| Day 3–7 | Open-answer tasks |
| Day 7–10 | Container, queue, API, robustness tests |
| Week 2–4 | Beta (Read Aloud + Repeat Sentence first), teacher spot checks |

## Not in scope yet

- Identical scores to Alfa on every answer; this is a practice estimate, not an official PTE score.
- Reading and Listening contributions from Speaking tasks.
