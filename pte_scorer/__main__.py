"""Command line interface.

  python -m pte_scorer score AUDIO --text "passage..." | --text-file passage.txt [--model PATH] [--json]
  python -m pte_scorer validate DATASET_DIR [--no-audio]
  python -m pte_scorer extract DATASET_DIR --out features.parquet [--cache DIR] [--limit N]
  python -m pte_scorer train FEATURES.parquet --out models/read_aloud.joblib [--group-by speaker_id]
  python -m pte_scorer serve [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pte_scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score one recording")
    s.add_argument("audio")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    s.add_argument("--model", default=None, help="trained model (.joblib); default models/read_aloud.joblib")
    s.add_argument("--json", action="store_true", help="print full JSON (features + feedback)")
    s.add_argument("--pte-type", choices=["academic", "core"], default="academic")

    v = sub.add_parser("validate", help="check a dataset against the data spec")
    v.add_argument("root")
    v.add_argument("--no-audio", action="store_true", help="skip decoding every audio file")

    e = sub.add_parser("extract", help="extract features for every response in a dataset")
    e.add_argument("root")
    e.add_argument("--out", required=True)
    e.add_argument("--cache", default=None)
    e.add_argument("--limit", type=int, default=None)

    t = sub.add_parser("train", help="train trait models from a feature table")
    t.add_argument("features")
    t.add_argument("--out", default="models/read_aloud.joblib")
    t.add_argument("--group-by", default="speaker_id", help="speaker_id or passage_id")
    t.add_argument("--folds", type=int, default=5)

    sv = sub.add_parser("serve", help="run the HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)

    if a.cmd == "score":
        from .pipeline import Analyzer
        from .scorer import DEFAULT_MODEL_PATH, Scorer

        text = a.text if a.text is not None else Path(a.text_file).read_text()
        analysis = Analyzer().analyze(a.audio, text)
        result = Scorer(a.model or DEFAULT_MODEL_PATH).score(analysis, a.pte_type)
        if a.json:
            print(json.dumps({**result, "features": analysis["features"]}, indent=2, ensure_ascii=False))
        else:
            _print_result(result)

    elif a.cmd == "validate":
        from .dataset import dataset_report, validate

        probs = validate(a.root, check_audio=not a.no_audio)
        if probs.empty:
            print("OK: no problems found")
        else:
            print(probs.to_string(index=False))
            print(f"\n{len(probs)} problem(s)")
        try:
            print("\n" + dataset_report(a.root))
        except Exception as ex:  # noqa: BLE001
            print(f"(report unavailable: {ex})")
        sys.exit(1 if not probs.empty else 0)

    elif a.cmd == "extract":
        from .dataset import extract_features

        cache = a.cache or str(Path("cache") / Path(a.root).name)
        extract_features(a.root, cache, a.out, limit=a.limit)

    elif a.cmd == "train":
        from .model import top_features, train

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        bundle = train(a.features, a.out, group_by=a.group_by, folds=a.folds)
        for trait in bundle["models"]:
            print(f"\nTop features for {trait}:")
            for name, share in top_features(bundle, trait, 10):
                print(f"  {name:28s} {share:.1%}")
        print(f"\nsaved {a.out}")

    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("pte_scorer.api:app", host=a.host, port=a.port)


def _print_result(r: dict):
    s, src = r["scores"], r["score_source"]
    print("\nPTE Read Aloud")
    est = r["estimate"]
    for t in ("content", "fluency", "pronunciation"):
        print(f"  {t:14s} {s[t]:.1f} / 5   ~{est['traits_10_90'][t]}/90   ({src[t]})")
    print(f"\n  Estimated level for this answer: ~{est['read_aloud_10_90']}/90 ({est['cefr']})")
    print(f"  ({est['note']})")
    fb = r["feedback"]
    if fb.get("transcript"):
        print(f"\nHeard: {fb['transcript']}")
    for k, label in (("omitted_words", "Omitted"), ("inserted_words", "Inserted")):
        if fb.get(k):
            print(f"{label}: {', '.join(fb[k])}")
    if fb.get("substituted_words"):
        print("Substituted: " + ", ".join(f"{x['expected']}→{x['said']}" for x in fb["substituted_words"]))
    if fb.get("mispronounced_words"):
        print("Check pronunciation:")
        for m in fb["mispronounced_words"]:
            print(f"  {m['word']:16s} {m['label']:8s} {m['message']}")
    al = r.get("alfa_style")
    if al:
        print(f"\nAlfa-style: content {al['speech_content']}, fluency {al['speech_fluency']}, "
              f"pronunciation {al['display_pronunciation']} ({al['speech_pronunciation']}), "
              f"speaking {al['speakingScore']}")
        print(f"  {al['answerCompare']}")
        print(f"  {al['userAnswer']}")
    if fb.get("summary"):
        print("\nTips:")
        for tip in fb["summary"]:
            print(f"  - {tip}")


if __name__ == "__main__":
    main()
