"""HTTP API.

POST /score  multipart: audio=<file>, text=<passage>   -> scores + feedback
GET  /health
"""
from __future__ import annotations

import os
import tempfile
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .pipeline import Analyzer
from .scorer import DEFAULT_MODEL_PATH, Scorer

app = FastAPI(title="PTE Read Aloud scorer")
_lock = threading.Lock()   # models are not thread-safe on one GPU: score one request at a time
_state: dict = {}


def _get():
    if not _state:
        _state["analyzer"] = Analyzer()
        _state["scorer"] = Scorer(os.environ.get("PTE_MODEL", DEFAULT_MODEL_PATH))
    return _state["analyzer"], _state["scorer"]


@app.get("/health")
def health():
    return {"ok": True, "trained_model": bool(_state.get("scorer") and _state["scorer"].bundle)}


@app.post("/score")
def score(audio: UploadFile = File(...), text: str = Form(...), pte_type: str = Form("academic"),
          include_features: bool = Form(False)):
    if pte_type not in ("academic", "core"):
        raise HTTPException(400, "pte_type must be 'academic' or 'core'")
    if len(text.split()) < 3:
        raise HTTPException(400, "text must be the full Read Aloud passage")
    suffix = os.path.splitext(audio.filename or "")[1] or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio.file.read())
        tmp.flush()
        with _lock:
            analyzer, scorer = _get()
            try:
                analysis = analyzer.analyze(tmp.name, text)
            except RuntimeError as e:
                raise HTTPException(400, str(e)) from e
            result = scorer.score(analysis, pte_type)
    if include_features:
        result["features"] = analysis["features"]
    return result
