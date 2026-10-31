"""Whisper ASR (mlx-whisper on Apple Silicon) with word timestamps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Nudges Whisper to keep fillers/repetitions instead of producing a clean transcript.
# Never pass the reference passage as the prompt: Whisper would "hear" the passage.
VERBATIM_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."


@dataclass
class AsrWord:
    word: str
    start: float
    end: float
    prob: float


def transcribe(wav: np.ndarray, model: str = DEFAULT_MODEL) -> tuple[str, list[AsrWord]]:
    import mlx_whisper

    res = mlx_whisper.transcribe(
        wav.astype(np.float32),
        path_or_hf_repo=model,
        language="en",
        word_timestamps=True,
        initial_prompt=VERBATIM_PROMPT,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    words = [
        AsrWord(w["word"].strip(), float(w["start"]), float(w["end"]), float(w.get("probability", 1.0)))
        for seg in res.get("segments", [])
        for w in seg.get("words", [])
    ]
    return res.get("text", "").strip(), words
