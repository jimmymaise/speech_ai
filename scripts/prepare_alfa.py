"""Convert crawled Read Aloud responses (out_answers/read_aloud/<qid>/...) to this project's dataset format.

  python scripts/prepare_alfa.py SRC_DIR OUT_DIR                 # metadata only: labels.csv, passages.csv
  python scripts/prepare_alfa.py SRC_DIR OUT_DIR --download N    # also fetch up to N audio files (resumable)
  python scripts/prepare_alfa.py SRC_DIR OUT_DIR --task repeat_sentence

--task picks the crawl folder; the reference text comes from the question field in REFERENCE_FIELD.
Open-answer tasks (describe image, retell lecture, ...) also get the keywords Alfa marked as [[MATCH]].

Labels are the source platform's automatic scores, not human or Pearson scores:
  content       <- speech_content          (0-90)
  fluency       <- speech_fluency          (0-90)
  pronunciation <- display_pronunciation   (0-90)
  pron_5        <- speech_pronunciation    ("x out of 5")
Word-level labels (CORRECT / AVERAGE / WRONG) and pause markers are parsed from
mobile_speech_transcript_with_pause into word_labels.jsonl.

Personal data: student_name and student_image_path are dropped; student_id is re-hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

REFERENCE_FIELD = {  # question.json field holding the text the answer is compared with
    "read_aloud": "qText", "repeat_sentence": "answer", "answer_short_question": "answer",
    "retell_lecture": "transcript", "respond_to_situation": "answer", "summarize_group_discussion": "qText",
    "describe_image": None,
}
MATCH = re.compile(r"\[\[MATCH\]\](.*?)\[\[/MATCH\]\]", re.S)
TAG = re.compile(r"\[\[?(CORRECT|AVERAGE|WRONG|PAUSE)\]\]?(.*?)\[\[?/\1\]\]?", re.S)


def clean_passage(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_word_labels(marked: str | None) -> list[list[str]]:
    """'[CORRECT]a b[/CORRECT] [[PAUSE]]/[[/PAUSE]] [[WRONG]]c[[/WRONG]]' -> [[a,CORRECT],[b,CORRECT],[/,PAUSE],[c,WRONG]]"""
    out = []
    for label, chunk in TAG.findall(marked or ""):
        if label == "PAUSE":
            out.append(["/", "PAUSE"])
        else:
            out.extend([w, label] for w in chunk.split())
    return out


def anon(student_id: str) -> str:
    return "s_" + hashlib.sha1(f"pte-ra:{student_id}".encode()).hexdigest()[:12]


def _json(v) -> dict:
    try:
        out = json.loads(v) if isinstance(v, str) else (v or {})
    except json.JSONDecodeError:
        return {}
    return out if isinstance(out, dict) else {}


def num(v):
    m = re.search(r"[\d.]+", str(v)) if v is not None else None
    return float(m.group()) if m else None


def convert(src: Path, out: Path, task: str = "read_aloud") -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows, passages, words = [], [], []
    qdirs = sorted(p for p in (src / task).iterdir() if p.is_dir())
    skipped = []
    for qd in qdirs:
        try:
            q = json.loads((qd / "question.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            skipped.append(qd.name)  # crawl still writing this question
            continue
        prefix = "ra" if task == "read_aloud" else "".join(w[0] for w in task.split("_"))
        pid = f"{prefix}_{q['id']}"
        field = REFERENCE_FIELD.get(task)
        passages.append({"passage_id": pid, "reference_text": clean_passage(q.get(field)) if field else "",
                         "difficulty": q.get("difficulty"), "tags": q.get("hashTags"),
                         "prompt_audio": q.get("qAudio"), "prompt_image": q.get("qImage")})
        for f in sorted(qd.glob("sample_response_*.json")):
            try:
                responses = json.loads(f.read_text())
            except json.JSONDecodeError:
                skipped.append(f"{qd.name}/{f.name}")
                continue
            if not isinstance(responses, list):   # {"same_as": ...}: the other file has the same answers
                continue
            for r in responses:
                if not r.get("audio_mp3_path"):
                    continue
                rid = f"alfa_{r['id']}"
                url = r["audio_mp3_path"]
                rows.append({
                    "response_id": rid, "passage_id": pid, "speaker_id": anon(str(r.get("student_id"))),
                    "audio_path": f"audio/{rid}.mp3", "audio_url": url,
                    "content": num(r.get("speech_content")), "fluency": num(r.get("speech_fluency")),
                    "pronunciation": num(r.get("display_pronunciation")),
                    "pron_5": num(r.get("speech_pronunciation")), "speaking": num(r.get("speakingScore")),
                    "listening": num(r.get("listeningScore")), "reading": num(r.get("readingScore")),
                    "pause_count": _json(r.get("answerCompare")).get("pause_count"),
                    "pte_type": r.get("pte_type"), "engine_version": r.get("version"),
                    "is_ai_evaluation": r.get("is_ai_evaluation"), "is_inflated": r.get("is_inflated"),
                    "submitted_at": r.get("submissionTime"), "source_file": f.name,
                })
                marked = r.get("marked_google_transcript") or ""
                words.append({"response_id": rid,
                              "words": parse_word_labels(r.get("mobile_speech_transcript_with_pause")
                                                         or r.get("userAnswer")),
                              "keywords": [m.strip().lower() for m in MATCH.findall(marked)] if marked else None})
    df = pd.DataFrame(rows).drop_duplicates("response_id")
    df.to_csv(out / "labels.csv", index=False)
    pd.DataFrame(passages).to_csv(out / "passages.csv", index=False)
    with open(out / "word_labels.jsonl", "w") as fh:
        for w in words:
            fh.write(json.dumps(w) + "\n")
    print(f"{len(qdirs) - sum('/' not in x for x in skipped)} passages, {len(df)} responses with audio, "
          f"{df.speaker_id.nunique()} speakers")
    if skipped:
        print(f"skipped (incomplete/unreadable): {skipped}")
    return df


def download(df: pd.DataFrame, out: Path, limit: int, workers: int = 4):
    """Resumable: files already on disk are skipped; partial files are written as .part first."""
    from concurrent.futures import ThreadPoolExecutor

    (out / "audio").mkdir(exist_ok=True)
    todo = [r for r in df.itertuples() if not (out / r.audio_path).exists()][:limit]

    def fetch(r):
        dst = out / r.audio_path
        for attempt in range(3):
            try:
                with urllib.request.urlopen(r.audio_url, timeout=30) as resp:
                    dst.with_suffix(".part").write_bytes(resp.read())
                dst.with_suffix(".part").rename(dst)
                return None
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(1 + attempt)
        return f"{r.response_id}: {err}"

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        for k, res in enumerate(ex.map(fetch, todo), 1):
            if res:
                failed.append(res)
            if k % 250 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} done, {len(failed)} failed", flush=True)
    for f in failed[:20]:
        print("failed", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--download", type=int, default=0, help="download up to N audio files")
    ap.add_argument("--selected", default=None, help="only download responses listed in this CSV (e.g. selected.csv)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--task", default="read_aloud", choices=sorted(REFERENCE_FIELD))
    a = ap.parse_args()
    df = convert(Path(a.src), Path(a.out), a.task)
    if a.download:
        if a.selected:
            df = df[df.response_id.isin(pd.read_csv(a.selected).response_id)]
        download(df, Path(a.out), a.download, a.workers)


if __name__ == "__main__":
    main()
