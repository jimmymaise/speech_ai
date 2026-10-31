# Output format

`Scorer.score(analysis, pte_type)` (CLI `score --json`, API `POST /score`) returns:

| Field | Content |
|---|---|
| `scores` | content / fluency / pronunciation on the PTE 0–5 trait scale |
| `scores_rounded` | the same, rounded |
| `score_source` | `model (raw 64/90)` or `rules` per trait |
| `estimate` | each trait on 10–90, the mean, a CEFR level (A1–C2) and a note that this is not a PTE score |
| `alfa_style` | the same fields as an Alfa Read Aloud result (below) |
| `words` | one entry per passage word, with label, timing and sound problems |
| `feedback` | tips, omitted / inserted / substituted words, mispronounced words, dropped ending sounds, mid-phrase pauses, transcript |

## `alfa_style`

```json
{
  "speech_content": 62,
  "speech_fluency": 47,
  "display_pronunciation": 64,
  "speech_pronunciation": "3 out of 5",
  "speakingScore": 57,
  "userAnswer": "[CORRECT]the first research study[/CORRECT] [[WRONG]]that[[/WRONG]] [CORRECT]has undergone[/CORRECT] … [[PAUSE]]//[[/PAUSE]] …",
  "answerCompare": {"pause_count": 1, "bad_word_count": "9 words", "average_word_count": "0 words",
                    "total_word_count": "63 words", "good_word_count": "54 words"},
  "transcript": "the first research study has undergone per review …"
}
```

- Scores are 0–90. No speech gives 0, like the source.
- `speakingScore` uses the per-type weights in `scorer.SPEAKING_WEIGHTS`.
- `speech_pronunciation` uses the bands in `scorer.PRON_5_BANDS`.
- `userAnswer` tags: `[CORRECT]…[/CORRECT]`, `[[AVERAGE]]…[[/AVERAGE]]`, `[[WRONG]]…[[/WRONG]]`,
  `[[PAUSE]]/[[/PAUSE]]` (one `/` per `pause_slash_s` seconds).

## `words`

```json
{"index": 12, "word": "asked", "label": "WRONG", "status": "read", "start": 4.12, "end": 4.48,
 "issues": [{"sound": "t", "position": "ending", "gop": -7.1, "problem": "missing", "ending_consonant": true}],
 "message": "ending sound /t/ missing"}
```

| Field | Values |
|---|---|
| `label` | `CORRECT` / `AVERAGE` / `WRONG` (word classifier trained on the source's word labels; GOP thresholds if no classifier) |
| `status` | `read`, `substituted` (+ `said`), `omitted` |
| `issues[].position` | `start`, `middle`, `ending` |
| `issues[].problem` | `replaced` (+ `heard`: the sound it sounded like) or `missing` (sound not heard and nothing else in its place) |
| `message` | short learner-facing text, e.g. `first sound /θ/ sounded like /t/`, `read as 'economy'`, `not read` |

Not flagged as errors:

- weak forms of small words (`a` /ə/, `the` /ðə/, `to` /tə/, `and` /ən/, ...), see `words.FUNCTION_WORDS`;
- dropping the final consonant in `and`, `just`, `must`, `next`, `first`;
- any sound problem on a word the classifier judges `CORRECT`.

Word-level settings live in `pte_scorer/words.py` (`DEFAULTS`) and can be overridden by `models/word_labels.json`.
