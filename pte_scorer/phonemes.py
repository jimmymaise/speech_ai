"""Pronunciation: G2P, wav2vec2 phoneme posteriors, CTC forced alignment and GOP."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import SAMPLE_RATE

PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
FRAME_S = 0.02  # wav2vec2 frame stride (320 samples @ 16 kHz)
VOWEL_CHARS = set("aeiouæɑɒɔəɚɛɜɪʊʌᵻɐɝ")
# Final consonants Vietnamese speakers commonly drop or weaken.
FINAL_CONSONANTS = {"t", "d", "k", "ɡ", "p", "b", "s", "z", "v", "f", "θ", "ð", "ʃ", "ʒ", "tʃ", "dʒ", "l", "n", "m", "ŋ"}


def is_vowel(ph: str) -> bool:
    return any(c in VOWEL_CHARS for c in ph) or ph in ("l̩", "n̩", "əl")


# --------------------------------------------------------------------------- G2P
@lru_cache(maxsize=1)
def _vocab() -> dict[str, int]:
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(PHONE_MODEL).tokenizer.get_vocab()


# espeak allophones that learners are not expected to produce: use the citation form
_ESPEAK_FIXES = {"ʔ": "t"}
# syllabic consonants: the model emits them as schwa + consonant
_ESPEAK_SPLITS = {"n̩": ["ə", "n"], "l̩": ["əl"], "m̩": ["ə", "m"]}


def _fit_to_vocab(ph: str, vocab: dict[str, int]) -> list[str]:
    """Map an espeak phone onto the model's token set (greedy longest match)."""
    ph = _ESPEAK_FIXES.get(ph, ph)
    if ph in _ESPEAK_SPLITS:
        return list(_ESPEAK_SPLITS[ph])
    if ph in vocab:
        return [ph]
    out, i = [], 0
    while i < len(ph):
        for j in range(len(ph), i, -1):
            if ph[i:j] in vocab:
                out.append(ph[i:j])
                i = j
                break
        else:
            i += 1  # drop unknown diacritic
    return out


_PHONE_CACHE: dict[str, tuple[str, ...]] = {}


@lru_cache(maxsize=1)
def _espeak():
    from phonemizer.backend import EspeakBackend
    return EspeakBackend("en-us", preserve_punctuation=False, with_stress=False)


def prefetch_phones(words: list[str]) -> None:
    """Phonemise all unseen words in one espeak call (much faster than one call per word)."""
    from phonemizer.separator import Separator

    new = sorted({w for w in words if w and w not in _PHONE_CACHE})
    if not new:
        return
    out = _espeak().phonemize(new, separator=Separator(phone=" ", word="|", syllable=""), strip=True)
    vocab = _vocab()
    for w, s in zip(new, out):
        phones: list[str] = []
        for p in s.replace("|", " ").split():
            phones.extend(_fit_to_vocab(p, vocab))
        _PHONE_CACHE[w] = tuple(phones)


def word_phones(word: str) -> tuple[str, ...]:
    if word not in _PHONE_CACHE:
        prefetch_phones([word])
    return _PHONE_CACHE[word]


# ------------------------------------------------------------------ acoustic model
class PhoneModel:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(PHONE_MODEL)
        self.model = AutoModelForCTC.from_pretrained(PHONE_MODEL).eval()
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        vocab = self.proc.tokenizer.get_vocab()
        self.vocab = vocab
        self.id2tok = {i: t for t, i in vocab.items()}
        self.blank = self.proc.tokenizer.pad_token_id
        special = {"<pad>", "<s>", "</s>", "<unk>"}
        self.phone_ids = np.array(sorted(i for t, i in vocab.items() if t not in special))

    def log_probs(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = self.proc(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(x).logits[0].float()
        return torch.log_softmax(logits, -1).cpu().numpy()

    def greedy_decode(self, lp: np.ndarray) -> list[str]:
        ids = lp.argmax(-1)
        out, prev = [], -1
        for i in ids:
            if i != prev and i != self.blank:
                out.append(self.id2tok[int(i)])
            prev = i
        return out


# --------------------------------------------------------------- forced alignment
def ctc_forced_align(lp: np.ndarray, tokens: list[int], blank: int) -> np.ndarray | None:
    """Viterbi CTC alignment. Returns, per frame, the index of the aligned token or -1 for blank."""
    T, L = lp.shape[0], len(tokens)
    if L == 0:
        return np.full(T, -1)
    S = 2 * L + 1
    ext = np.full(S, blank)
    ext[1::2] = tokens
    # skip transition s-2 -> s allowed for non-blank states whose token differs from s-2
    can_skip = np.zeros(S, bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = lp[0, blank]
    alpha[1] = lp[0, ext[1]]
    bp = np.zeros((T, S), np.int8)
    emit = lp[:, ext]  # (T, S)
    for t in range(1, T):
        stay = alpha
        prev1 = np.concatenate(([NEG], alpha[:-1]))
        prev2 = np.where(can_skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        stacked = np.stack([stay, prev1, prev2])
        choice = stacked.argmax(0)
        alpha = stacked[choice, np.arange(S)] + emit[t]
        bp[t] = choice
    end = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[end] < NEG / 2:
        return None  # audio too short for the token sequence
    path = np.empty(T, np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return np.where(path % 2 == 1, path // 2, -1)


# ------------------------------------------------------------------------- GOP
@dataclass
class PhoneScore:
    phone: str
    word_idx: int          # index into the aligned word list
    start: float
    end: float
    gop: float             # <= 0; ~0 = canonical, very negative = likely mispronounced
    lpp: float             # mean log posterior of the canonical phone
    competitor: str        # most likely other phone where this one should be
    pmax: float = 1.0      # highest posterior of this phone between its neighbours; ~0 = not said at all
    comp_pmax: float = 0.0  # same for the competitor: high = replaced by it, low too = probably dropped


@dataclass
class WordAlignment:
    word: str
    ref_idx: int | None    # None for inserted words
    start: float
    end: float
    phones: list[PhoneScore] = field(default_factory=list)


def score_phones(pm: PhoneModel, lp: np.ndarray, words: list[tuple[str, int | None]]
                 ) -> tuple[list[WordAlignment], list[PhoneScore]] | None:
    """Force-align the given word sequence and compute GOP for every phone."""
    seq, owner = [], []
    for wi, (w, _) in enumerate(words):
        for ph in word_phones(w):
            seq.append(ph)
            owner.append(wi)
    tok = [pm.vocab[p] for p in seq]
    frame_tok = ctc_forced_align(lp, tok, pm.blank)
    if frame_tok is None:
        return None

    # renormalised posteriors over phones only (blank removed): GOP is not dominated by blank
    plp = lp[:, pm.phone_ids]
    plp = plp - np.logaddexp.reduce(plp, axis=1, keepdims=True)
    col = {int(t): c for c, t in enumerate(pm.phone_ids)}
    p_blank = np.exp(lp[:, pm.blank])

    phones: list[PhoneScore] = []
    for k, ph in enumerate(seq):
        frames = np.where(frame_tok == k)[0]
        c = col[tok[k]]
        f = plp[frames]
        w = 1.0 - p_blank[frames] + 1e-3
        gop = float(np.sum(w * (f[:, c] - f.max(1))) / np.sum(w))
        lpp = float(lp[frames, tok[k]].mean())
        f2 = f.copy()
        f2[:, c] = -np.inf
        comp = pm.id2tok[int(pm.phone_ids[f2.max(0).argmax()])]
        # search window: from the previous phone to the next one (a deleted phone gets squeezed in there)
        prev_f = np.where(frame_tok == k - 1)[0] if k > 0 else np.array([0])
        next_f = np.where(frame_tok == k + 1)[0] if k + 1 < len(seq) else np.array([len(lp) - 1])
        lo, hi = int(prev_f[0]) if prev_f.size else frames[0], int(next_f[-1]) if next_f.size else frames[-1]
        pmax = float(np.exp(lp[lo:hi + 1, tok[k]].max()))
        comp_pmax = float(np.exp(lp[frames[0]:frames[-1] + 1, pm.vocab[comp]].max()))
        phones.append(PhoneScore(ph, owner[k], frames[0] * FRAME_S, (frames[-1] + 1) * FRAME_S, gop, lpp, comp,
                                 pmax, comp_pmax))

    aligned: list[WordAlignment] = []
    for wi, (w, ridx) in enumerate(words):
        ps = [p for p in phones if p.word_idx == wi]
        if ps:
            aligned.append(WordAlignment(w, ridx, ps[0].start, ps[-1].end, ps))
    return aligned, phones


def phone_error_rate(ref: list[str], hyp: list[str]) -> float:
    """Edit distance between canonical phones and what the model freely decoded."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m] / n
