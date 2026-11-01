"""Answer Short Question: content = 1 iff any single word of the answer key is in APEUni's transcript.

    python scripts/apeuni/asq.py
"""
import re

from common import questions


def key_words(prediction_text):
    return {w for alt in prediction_text.split("Answer:")[1].split("/") for w in alt.lower().split()}


def main():
    for q, qj, answers in questions("answer_short_question"):
        words = key_words(qj["prediction_text"])
        agree = 0
        for x in answers:
            a = x["attributes"]
            text = ((a.get("score_details") or {}).get("audio_text") or "").lower()
            hit = bool(set(re.findall(r"[a-z0-9']+", text)) & words)
            agree += hit == bool(int(a["scores"]["components"]["content"]["score"]))
        print(f"{q} key {sorted(words)}: rule agrees on {agree}/{len(answers)}")


if __name__ == "__main__":
    main()
