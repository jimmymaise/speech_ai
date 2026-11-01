"""Repeat Sentence content: fit the penalty of each sentence word (missed or out of order) per question.

    python scripts/apeuni/rs_content.py
"""
import numpy as np
from sklearn.linear_model import LinearRegression

from common import questions


def main():
    for q, qj, answers in questions("repeat_sentence"):
        words = [t["target"] for t in answers[0]["attributes"]["score_details"]["details"]]
        X, y = [], []
        for x in answers:
            det = x["attributes"]["score_details"]["details"]
            if len(det) == len(words):
                X.append([t["type"] != "content_word" for t in det])
                y.append(float(x["attributes"]["scores"]["components"]["content"]["score"]))
        X, y = np.array(X, float), np.array(y)
        few = X.sum(1) <= 3
        m = LinearRegression().fit(X[few], y[few])
        print(f"{q} {qj['prediction_text']}")
        print(f"   intercept {m.intercept_:.2f}, MAE (<=3 missed) {np.abs(m.predict(X[few]) - y[few]).mean():.3f}, "
              f"MAE (all) {np.abs(m.predict(X) - y).mean():.3f}")
        print("   " + "  ".join(f"{w}:{-c:.2f}" for w, c in zip(words, m.coef_)))


if __name__ == "__main__":
    main()
