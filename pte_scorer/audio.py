"""Audio loading: any format ffmpeg understands -> float32 mono 16 kHz."""
from __future__ import annotations

import subprocess

import numpy as np

from . import SAMPLE_RATE


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode with ffmpeg. No trimming or normalisation: silence is a fluency signal."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
        "-f", "s16le", "-ac", "1", "-ar", str(sr), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {out.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(out.stdout, np.int16).astype(np.float32) / 32768.0


def audio_quality(wav: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return {
        "duration_s": wav.size / sr,
        "peak": peak,
        "clipping_ratio": float(np.mean(np.abs(wav) > 0.99)) if wav.size else 0.0,
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)) if wav.size else -180.0,
    }
