"""Agreement, accuracy and paired-comparison statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def agreement(a, b) -> dict:
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    return {"agreement": float(np.mean(a == b)), "kappa": float(cohen_kappa_score(a.astype(str), b.astype(str)))}


def accuracy(y_true, y_pred, positive: str | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}
    if positive is not None:
        out["f1_pos"] = float(f1_score(y_true, y_pred, pos_label=positive, average="binary", zero_division=0))
    return out


def bootstrap_ci(fn, *arrays, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap interval of a statistic computed on paired arrays."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a, dtype=object) for a in arrays]
    n = len(arrays[0])
    point = fn(*arrays)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(fn(*(a[idx] for a in arrays)))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on the discordant pairs of two classifiers."""
    y, a, b = (np.asarray(x).astype(str) for x in (y_true, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}
