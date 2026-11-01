"""Collect the source's per-answer outputs for an open-answer task (scores, transcript, keyword matches, pause marks).

    python scripts/collect_open_alfa.py $CRAWL_DIR describe_image data/alfa_di_open.parquet

Note: crawl folders respond_to_situation/ and summarize_group_discussion/ are swapped (see docs/alfa_formulas.md).
"""
import json, re, sys
from pathlib import Path
import pandas as pd
crawl = Path(sys.argv[1]); task = sys.argv[2]; out = sys.argv[3]
MATCH = re.compile(r"\[\[MATCH\]\](.*?)\[\[/MATCH\]\]")
PAUSE = re.compile(r"\[\[PAUSE\]\](/*)\[\[/PAUSE\]\]")
TOK = re.compile(r"[A-Za-z0-9']+")
def num(v):
    try: return float(str(v).split()[0])
    except Exception: return None
rows = []
for qd in (crawl / task).iterdir():
    if not qd.is_dir(): continue
    q = {}
    try: q = json.load(open(qd / "question.json"))
    except Exception: pass
    for fn in ("sample_response_ac.json", "sample_response_core.json"):
        p = qd / fn
        if not p.exists(): continue
        try: x = json.load(open(p))
        except Exception: continue
        if isinstance(x, dict) and "same_as" in x: continue
        rs = x if isinstance(x, list) else (x.get("data") or x.get("responses") or [])
        if not isinstance(rs, list): continue
        for r in rs:
            if not r.get("audio_mp3_path"): continue
            g = r.get("google_transcript") or ""
            mk = r.get("marked_google_transcript") or ""
            mob = r.get("mobile_speech_transcript_with_pause") or ""
            ua = r.get("userAnswer") or ""
            m = [w.strip().lower() for w in MATCH.findall(mk or mob)]
            pz = PAUSE.findall(mob)
            rows.append(dict(rid=r["id"], qid=qd.name, file=fn, pte_type=r.get("pte_type"), ver=r.get("version"),
                content=num(r.get("speech_content")), fluency=num(r.get("speech_fluency")),
                pron=num(r.get("display_pronunciation")), pron5=num(r.get("speech_pronunciation")),
                speaking=num(r.get("speakingScore")), listening=num(r.get("listeningScore")),
                inflated=r.get("is_inflated"), n_words=len(TOK.findall(g)), n_words_ua=len(TOK.findall(re.sub(r"\[\[?/?[A-Z]+\]\]?", " ", ua))),
                n_match=len(m), n_match_distinct=len(set(m)), has_mobile=bool(mob), n_pause=len(pz),
                n_slash=sum(len(s) for s in pz), ua=ua[:400], g=g, matched=" ".join(sorted(set(m)))))
d = pd.DataFrame(rows).drop_duplicates("rid")
d.to_parquet(out); print(task, len(d))
