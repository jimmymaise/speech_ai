# APEUni analysis scripts

Reproduce the findings in [docs/apeuni_formulas.md](../../docs/apeuni_formulas.md) from a decoded APEUni crawl.

| Variable | Default | Meaning |
|---|---|---|
| `APE_CRAWL_DIR` | `data/ape_crawl` | `<task>/<question_id>/{answers.json, questions.json}` |
| `APE_OUT_DIR` | `cache/apeuni` | parquet tables written by the scripts |
| `APE_AUDIO_DIR` | `data/ape_audio` | answer audio, `<task>/<question_id>/<answer_id>.wav` |

| Script | What it does |
|---|---|
| `../apeuni_formulas.py <read_aloud dir>` | Read Aloud: sound / word score rules, task score, content, fluency and pronunciation fits |
| `flatten.py <task>` | one row per answer for `repeat_sentence`, `describe_image`, `retell_lecture`, `summarize_group_discussion`, `respond_situation` |
| `fit.py <task>` | checks the task score rule, fits pronunciation, fluency and content (whole questions held out) |
| `rs_content.py` | Repeat Sentence: penalty of each sentence word, per question |
| `asq.py` | Answer Short Question: any-answer-key-word rule against every answer |
| `audio.py download\|features <task> <ids...>` | resumable audio download; pause features with the scorer's energy VAD |

```bash
export APE_CRAWL_DIR=/path/to/ape_crawler
python scripts/apeuni_formulas.py $APE_CRAWL_DIR/read_aloud
python scripts/apeuni/flatten.py describe_image && python scripts/apeuni/fit.py describe_image
python scripts/apeuni/asq.py
python scripts/apeuni/audio.py download repeat_sentence 3407 && python scripts/apeuni/audio.py features repeat_sentence 3407
```

Crawled data, audio and outputs stay out of git (`data/`, `cache/`). Student nicknames in the crawl are never written
to the outputs.
