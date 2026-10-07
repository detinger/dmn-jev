"""Calibration of System One probabilities against ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def top_label(proba: pd.DataFrame, y_true) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(confidence, predicted class, correctness) for the top-label view of a distribution."""
    conf = proba.max(axis=1).to_numpy()
    pred = proba.idxmax(axis=1).to_numpy()
    return conf, pred, pred == np.asarray(y_true)


def reliability(conf, correct, n_bins: int = 15, strategy: str = "uniform") -> pd.DataFrame:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "confidence": conf[m].mean(), "accuracy": correct[m].mean()})
    return pd.DataFrame(rows)


def ece(conf, correct, n_bins: int = 15, strategy: str = "uniform") -> float:
    r = reliability(conf, correct, n_bins, strategy)
    return float((r["n"] * (r["confidence"] - r["accuracy"]).abs()).sum() / r["n"].sum())


def mce(conf, correct, n_bins: int = 15) -> float:
    r = reliability(conf, correct, n_bins)
    return float((r["confidence"] - r["accuracy"]).abs().max())


def brier(proba: pd.DataFrame, y_true) -> float:
    """Multi-class Brier score (sum over classes, averaged over cases)."""
    onehot = pd.get_dummies(pd.Categorical(np.asarray(y_true), categories=proba.columns)).to_numpy(float)
    return float(np.mean(np.sum((proba.to_numpy(float) - onehot) ** 2, axis=1)))


class TopLabelIsotonic:
    """Post-hoc isotonic recalibration of the top-label confidence (Zadrozny & Elkan, 2002)."""

    def fit(self, conf, correct) -> "TopLabelIsotonic":
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, np.asarray(correct, float))
        return self

    def transform(self, conf) -> np.ndarray:
        return self.iso.predict(np.asarray(conf, float))


def summary(proba: pd.DataFrame, y_true, calibrator: TopLabelIsotonic | None = None) -> dict:
    conf, _, correct = top_label(proba, y_true)
    out = {"accuracy": float(correct.mean()), "ece": ece(conf, correct), "ece_quantile": ece(conf, correct, strategy="quantile"),
           "mce": mce(conf, correct), "brier": brier(proba, y_true), "mean_confidence": float(conf.mean())}
    if calibrator is not None:
        c2 = calibrator.transform(conf)
        out["ece_isotonic"] = ece(c2, correct)
        out["ece_isotonic_quantile"] = ece(c2, correct, strategy="quantile")
    return out
