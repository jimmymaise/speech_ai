# PTE Speaking scorer

Scores PTE Speaking answers on the three traits **Content, Oral Fluency, Pronunciation** (0–5 and 0–90),
with word-level feedback: omitted / inserted / substituted words, mispronounced sounds (including dropped
ending sounds), and pauses in the middle of a phrase. Output also comes in Alfa's result format
(`alfa_style`: scores, "x out of 5", speaking score, `[CORRECT]` / `[[WRONG]]` / `[[PAUSE]]` tagged answer).

Read Aloud works end to end; Repeat Sentence uses the same pipeline. The other Speaking tasks are in progress,
see [docs/plan.md](docs/plan.md).

Runs offline on Apple Silicon (`mlx-whisper`). A `faster-whisper` backend for Linux GPU / CPU servers is planned.

| Doc | Content |
|---|---|
| [docs/plan.md](docs/plan.md) | scope, quality targets, compute and serving cost, timeline |
| [docs/data_pipeline.md](docs/data_pipeline.md) | crawl → sample → download → extract → train → evaluate |
| [docs/output_format.md](docs/output_format.md) | every field of the scoring result |
| [docs/alfa_formulas.md](docs/alfa_formulas.md) | how the source scores each task: recovered formulas and open questions |
| [docs/apeuni_formulas.md](docs/apeuni_formulas.md) | how APEUni scores all seven Speaking tasks (separate from the source above) |
| [docs/read_aloud_design.md](docs/read_aloud_design.md) | Read Aloud v2 design: what to keep and avoid from Alfa and APEUni, components, output modes, validation |

## How it works

```
audio ──ffmpeg──► 16 kHz mono
passage ──normalise──► words (+ where punctuation allows a pause) ──espeak G2P──► phones
          │
          ├─ Whisper large-v3-turbo (mlx) ─► transcript ─► word alignment vs passage ─► CONTENT features
          ├─ wav2vec2 phoneme CTC ─► forced alignment of the spoken words ─► GOP per phone ─► PRONUNCIATION features
          └─ energy VAD + alignment timings + Praat pitch ─► pauses / rate / rhythm / intonation ─► FLUENCY features
                                         │
                       ~55 features ─► LightGBM per trait (if trained) or transparent rules
```

| Module | What it does |
|---|---|
| `pte_scorer/audio.py` | ffmpeg decode, audio quality stats |
| `pte_scorer/text.py` | number expansion, tokenisation, punctuation-pause map |
| `pte_scorer/asr.py` | Whisper with a verbatim prompt (keeps "um", repetitions) |
| `pte_scorer/content.py` | weighted word alignment → omissions / insertions / substitutions / fillers / repetitions |
| `pte_scorer/phonemes.py` | espeak G2P, `facebook/wav2vec2-lv-60-espeak-cv-ft`, CTC Viterbi alignment, GOP |
| `pte_scorer/fluency.py` | pauses (at punctuation vs mid-phrase), speech/articulation rate, run length, pitch |
| `pte_scorer/pipeline.py` | glue: audio + passage → feature dict + details |
| `pte_scorer/scorer.py` | trained model or rules → scores, Alfa-style result, learner feedback |
| `pte_scorer/words.py` | per-word CORRECT / AVERAGE / WRONG, sound problems, Alfa-style tagged answer |
| `pte_scorer/levels.py` | 10–90 scale and CEFR level estimate |
| `pte_scorer/model.py` | LightGBM training, speaker-grouped CV, metrics (PCC, MAE, QWK, within ±1) |
| `pte_scorer/dataset.py` | dataset validation, report, resumable feature extraction |
| `pte_scorer/api.py` | FastAPI `POST /score` |

## Setup

```bash
brew install ffmpeg espeak-ng
~/.pyenv/versions/3.11.13/bin/python -m venv .venv
.venv/bin/pip install -r requirements.txt
```
First run downloads ~1.3 GB wav2vec2 + ~1.6 GB Whisper turbo.

## Usage

```bash
# score one recording
.venv/bin/python -m pte_scorer score answer.mp3 --text-file passage.txt
.venv/bin/python -m pte_scorer score answer.mp3 --text "Climate change is ..." --json

# HTTP API
.venv/bin/python -m pte_scorer serve --port 8000
curl -X POST localhost:8000/score -F audio=@answer.mp3 -F text="Climate change is ..."
```

## Your own data

Dataset folder layout:

```
my_data/
  audio/...                   any format ffmpeg reads; do NOT trim silence
  labels.csv                  response_id, passage_id, speaker_id, audio_path, content, fluency, pronunciation [, flags, rater_id, device]
  passages.csv                passage_id, reference_text
```

```bash
.venv/bin/python -m pte_scorer validate my_data                          # finds problems + prints counts/score distribution
.venv/bin/python -m pte_scorer extract  my_data --out cache/my.parquet   # resumable; ~1.2 s per clip on M4 Pro
.venv/bin/python -m pte_scorer train    cache/my.parquet --out models/read_aloud.joblib
.venv/bin/python -m pte_scorer train    cache/my.parquet --out /tmp/x.joblib --group-by passage_id   # unseen-passage check
```
`models/read_aloud.joblib` is picked up automatically by `score` and the API.
Traits without a trained model fall back to rules (`score_source` says which).

## Crawled practice-platform data

```bash
python scripts/prepare_alfa.py $CRAWL_DIR data/alfa --task read_aloud     # convert
python scripts/sample_alfa.py data/alfa --per-band 6 --core-share 0.5     # balanced selection + splits
CRAWL_DIR=... scripts/run_round.sh read_aloud data/alfa data/alfa_r1 cache/alfa_r1   # download + extract
python -m pte_scorer train cache/alfa_r1_features.parquet --out models/read_aloud.joblib
python scripts/eval_holdout.py cache/alfa_r1_features.parquet --curve      # held-out report
python scripts/train_word_labels.py data/alfa_r1 cache/alfa_r1             # word colour classifier
```
Details and the facts recovered from the data are in [docs/data_pipeline.md](docs/data_pipeline.md).
Data, audio, feature caches and trained models are not in the repo.

## Public stand-in data

`scripts/prepare_speechocean.py` downloads speechocean762 (5,000 L2 utterances, 250 speakers) and converts it
to the format above (0–10 scores halved to 0–5: completeness→content, fluency→fluency, accuracy→pronunciation).
Its content labels are almost all 5, so it only exercises the fluency and pronunciation models.
Its sentences are short (~3 s) and not PTE passages, so treat its model as a pipeline check / pretraining, not a PTE model.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Known limitations / next steps

- Rules are a fallback for traits without a trained model.
- Whisper can "auto-correct" a mispronounced word into the right one: content can be over-credited; GOP is the
  independent check.
- Level B (phone-level Transformer pretrained on speechocean762, fine-tuned on PTE data) is the planned next model.
