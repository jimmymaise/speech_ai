"""Shared paths and loaders for the APEUni analysis scripts.

Crawl layout: $APE_CRAWL_DIR/<task>/<question_id>/{answers.json, questions.json}
Outputs (parquet tables) go to $APE_OUT_DIR (default cache/apeuni).
"""
import json
import os

CRAWL = os.environ.get("APE_CRAWL_DIR", "data/ape_crawl")
OUT = os.environ.get("APE_OUT_DIR", "cache/apeuni")
AUDIO = os.environ.get("APE_AUDIO_DIR", "data/ape_audio")

STOP = set("""a an the and or but of to in on at for with by from as is are was were be been it its this that these
those there their they he she we you i his her our your not no so do does did can could will would should may might
also more most very than then which who whom what when where how about into over such some any each other all both
many much""".split())


def questions(task):
    root = os.path.join(CRAWL, task)
    for q in sorted(d for d in os.listdir(root) if d.isdigit()):
        a = os.path.join(root, q, "answers.json")
        if os.path.exists(a):
            yield int(q), json.load(open(os.path.join(root, q, "questions.json"))), json.load(open(a))["answers"]


def out_path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)
