"""Download APEUni answer audio and measure pauses with the scorer's energy VAD.

    python scripts/apeuni/audio.py download <task> <question_id> [...]
    python scripts/apeuni/audio.py features <task> <question_id> [...]

Downloads are resumable (existing files are skipped). Features: leading / trailing silence, speaking span, speech
time, number and length of inner pauses, longest pause.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import AUDIO, CRAWL, out_path  # noqa: E402
from pte_scorer.audio import load_audio  # noqa: E402
from pte_scorer.fluency import HOP_S, _runs, speech_mask  # noqa: E402

import json  # noqa: E402


def answers(task, q):
    return json.load(open(os.path.join(CRAWL, task, str(q), "answers.json")))["answers"]


def download(task, q):
    folder = os.path.join(AUDIO, task, str(q))
    os.makedirs(folder, exist_ok=True)
    jobs = []
    for x in answers(task, q):
        a = x["attributes"]
        path = os.path.join(folder, f"{a['id']}.wav")
        if a.get("audio_url") and not os.path.exists(path):
            jobs.append((a["audio_url"], path))

    def get(job):
        url, path = job
        r = subprocess.run(["curl", "-sf", "--retry", "3", "-o", path + ".part", url])
        if r.returncode == 0:
            os.replace(path + ".part", path)
        return r.returncode == 0

    with ThreadPoolExecutor(8) as pool:
        ok = sum(pool.map(get, jobs))
    print(f"{task}/{q}: {ok}/{len(jobs)} downloaded, {len(os.listdir(folder))} files")


def features(task, q):
    rows = []
    for x in answers(task, q):
        a = x["attributes"]
        path = os.path.join(AUDIO, task, str(q), f"{a['id']}.wav")
        if not os.path.exists(path):
            continue
        wav = load_audio(path)
        dur = len(wav) / 16000
        m = speech_mask(wav)
        sp, si = _runs(m, True), _runs(m, False)
        if sp:
            s0, s1 = sp[0][0] * HOP_S, sp[-1][1] * HOP_S
            inner = np.array([(b - e) * HOP_S for e, b in si if e * HOP_S > s0 and b * HOP_S < s1] or [0.0])
        else:
            s0 = s1 = 0.0
            inner = np.array([0.0])
        rows.append(dict(id=int(a["id"]), q=q, rec_s=dur, lead_s=s0, trail_s=dur - s1, span_s=s1 - s0,
                         speech_s=m.sum() * HOP_S, n_p02=int((inner >= .2).sum()), n_p03=int((inner >= .3).sum()),
                         n_p05=int((inner >= .5).sum()), pause_s=inner[inner >= .2].sum(), max_pause=inner.max()))
    path = out_path(f"audio_{task}_{q}.parquet")
    pd.DataFrame(rows).to_parquet(path)
    print(f"{task}/{q}: {len(rows)} answers -> {path}")


if __name__ == "__main__":
    action, task, qs = sys.argv[1], sys.argv[2], sys.argv[3:]
    for q in qs:
        (download if action == "download" else features)(task, int(q))
