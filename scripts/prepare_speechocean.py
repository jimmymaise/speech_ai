"""Download speechocean762 and convert it to this project's dataset format.

Output (data/speechocean762/):
  audio/<response_id>.wav
  labels.csv    response_id, passage_id, speaker_id, audio_path, content, fluency, pronunciation, split, ...
  passages.csv  passage_id, reference_text

speechocean762 scores are 0-10; they are halved onto the 0-5 PTE trait scale as a stand-in:
  content <- completeness, fluency <- fluency, pronunciation <- accuracy.
"""
import hashlib
from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset

OUT = Path("data/speechocean762")


def main():
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("mispeech/speechocean762").cast_column("audio", Audio(decode=False))
    rows, passages = [], {}
    for split in ("train", "test"):
        for i, r in enumerate(ds[split]):
            rid = f"{split}_{i:05d}"
            wav = OUT / "audio" / f"{rid}.wav"
            if not wav.exists():
                wav.write_bytes(r["audio"]["bytes"])
            text = r["text"].strip()
            pid = "so_" + hashlib.md5(text.lower().encode()).hexdigest()[:8]
            passages[pid] = text
            rows.append({
                "response_id": rid, "passage_id": pid, "speaker_id": f"so_{r['speaker']}",
                "audio_path": str(wav.relative_to(OUT)),
                "content": r["completeness"] / 2, "fluency": r["fluency"] / 2,
                "pronunciation": r["accuracy"] / 2,
                "prosodic": r["prosodic"] / 2, "total": r["total"] / 2,
                "gender": r["gender"], "age": r["age"], "split": split,
            })
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)
    pd.DataFrame({"passage_id": list(passages), "reference_text": list(passages.values())}
                 ).to_csv(OUT / "passages.csv", index=False)
    print(f"{len(rows)} responses, {len(passages)} passages, {pd.DataFrame(rows).speaker_id.nunique()} speakers")


if __name__ == "__main__":
    main()
