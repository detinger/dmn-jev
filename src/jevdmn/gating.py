"""Confidence gate: which cases may be decided automatically, with a bound on the error rate.

Thresholds are chosen with Learn-then-Test (Angelopoulos et al.) using fixed-sequence
testing and exact binomial p-values: for candidate thresholds ordered from the most
to the least conservative, the null "selective risk > eps" is tested on the
calibration set and rejected while p <= delta. The selected threshold controls
the error rate of automated cases at level eps with probability at least 1 - delta,
whether or not the scores are calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.tree import DecisionTreeClassifier


def risk_coverage(score, correct) -> pd.DataFrame:
    """Selective risk at every coverage level, accepting cases in order of decreasing score."""
    score, correct = np.asarray(score, float), np.asarray(correct, bool)
    order = np.argsort(-score, kind="stable")
    errs = np.cumsum(~correct[order])
    n = np.arange(1, len(score) + 1)
    return pd.DataFrame({"threshold": score[order], "coverage": n / len(score), "risk": errs / n})


def aurc(score, correct) -> float:
    return float(risk_coverage(score, correct)["risk"].mean())


@dataclass
class Threshold:
    value: float  # accept when score >= value (inf = accept nothing)
    eps: float
    delta: float
    calib_coverage: float
    calib_risk: float


def min_accept_for(eps: float, delta: float) -> int:
    """Smallest number of accepted cases for which zero errors rejects 'risk > eps' at level delta."""
    return int(np.ceil(np.log(delta) / np.log(1 - eps)))


def _thin(lams: np.ndarray, k: int) -> np.ndarray:
    """At most k thresholds, evenly spaced along the ordered sequence (first one kept)."""
    if len(lams) <= k:
        return lams
    return lams[np.unique(np.linspace(0, len(lams) - 1, k).round().astype(int))]


def candidate_thresholds(score_a, correct_a, eps: float, delta: float, n_b: int, max_candidates: int = 40) -> np.ndarray:
    """Ordered candidate thresholds (most conservative first), fixed by calibration split A only.

    The sequence starts at the most conservative threshold that a pilot test on split A
    already accepts (binomial test at level delta); if none does, at the first threshold
    that accepts enough cases to be testable on split B (min_accept_for(eps, delta), scaled).
    Starting too early wastes the single stop of fixed-sequence testing on tiny samples.
    """
    score_a, correct_a = np.asarray(score_a, float), np.asarray(correct_a, bool)
    fin = np.isfinite(score_a)
    lams = np.unique(score_a[fin])[::-1]
    if len(lams) == 0:
        return lams
    need = min_accept_for(eps, delta) * len(score_a) / max(n_b, 1)
    for k, lam in enumerate(lams):
        acc = score_a >= lam
        n, e = int(acc.sum()), int((~correct_a[acc]).sum())
        if n >= need and binom.cdf(e, n, eps) <= delta:
            return _thin(lams[k:], max_candidates)
    counts = np.array([(score_a >= lam).sum() for lam in lams])
    return _thin(lams[counts >= need], max_candidates)


def ltt_threshold(score, correct, eps: float, delta: float = 0.05, candidates=None) -> Threshold:
    """Fixed-sequence Learn-then-Test on calibration split B.

    `candidates` must not depend on (score, correct): pass thresholds derived from a
    separate split A (see candidate_thresholds). Each hypothesis H_j: R(lambda_j) > eps is
    tested with the exact binomial p-value P(Bin(n_j, eps) <= k_j); testing stops at the
    first non-rejection, which controls the family-wise error rate at delta.
    """
    score, correct = np.asarray(score, float), np.asarray(correct, bool)
    if candidates is None:
        candidates = np.unique(score[np.isfinite(score)])[::-1]
    chosen = np.inf
    for lam in candidates:
        acc = score >= lam
        n, k = int(acc.sum()), int((~correct[acc]).sum())
        p = binom.cdf(k, n, eps) if n else 1.0
        if p > delta:
            break
        chosen = lam
    acc = score >= chosen
    return Threshold(float(chosen), eps, delta, float(acc.mean()),
                     float((~correct[acc]).mean()) if acc.any() else 0.0)


def calibrate(score_a, correct_a, score_b, correct_b, eps: float, delta: float,
              critical_b=None) -> Threshold:
    """LTT threshold for the selective risk, and optionally also for the critical-class miss rate.

    With `critical_b` (True for calibration cases of the critical class), two families are tested on
    split B, each at delta/2 (Bonferroni): selective risk <= eps, and miss rate <= eps, where the miss
    rate is the share of critical-class cases that are decided automatically and wrongly. The more
    conservative of the two thresholds is returned, so both bounds hold jointly with probability >= 1 - delta.
    """
    cands = candidate_thresholds(score_a, correct_a, eps, delta, len(score_b))
    if critical_b is None:
        return ltt_threshold(score_b, correct_b, eps, delta, cands)
    sel = ltt_threshold(score_b, correct_b, eps, delta / 2, cands)
    miss = miss_rate_threshold(score_b, correct_b, critical_b, eps, delta / 2, cands)
    chosen = max(sel.value, miss)
    score_b, correct_b = np.asarray(score_b, float), np.asarray(correct_b, bool)
    acc = score_b >= chosen
    return Threshold(float(chosen), eps, delta, float(acc.mean()),
                     float((~correct_b[acc]).mean()) if acc.any() else 0.0)


def miss_rate_threshold(score, correct, critical, eps: float, delta: float, candidates) -> float:
    """Fixed-sequence LTT for the miss rate P(automated and wrong | critical class).

    The loss is computed over all n_c critical-class cases of split B (automated-and-wrong = 1), so
    H_j: miss(lambda_j) > eps is tested with P(Bin(n_c, eps) <= k_j). The miss rate grows as the
    threshold falls, so the conservative-first order of the candidates is the right testing order.
    At least min_accept_for(eps, delta) critical cases are needed to certify any threshold.
    """
    score, correct, critical = np.asarray(score, float), np.asarray(correct, bool), np.asarray(critical, bool)
    n_c = int(critical.sum())
    chosen = np.inf
    if n_c == 0:
        return chosen
    for lam in candidates:
        k = int((critical & (score >= lam) & ~correct).sum())
        if binom.cdf(k, n_c, eps) > delta:
            break
        chosen = lam
    return float(chosen)


def apply(score, thr: Threshold) -> np.ndarray:
    return np.asarray(score, float) >= thr.value


def selective_metrics(accept, correct, critical=None) -> dict:
    accept, correct = np.asarray(accept, bool), np.asarray(correct, bool)
    n_acc = int(accept.sum())
    out = {"coverage": float(accept.mean()), "selective_risk": float((~correct[accept]).mean()) if n_acc else 0.0,
           "n_auto": n_acc, "n_review": int((~accept).sum()),
           "overall_error_auto_only": float((~correct & accept).sum() / len(accept))}
    if critical is not None:
        critical = np.asarray(critical, bool)
        out["miss_rate"] = float((critical & accept & ~correct).sum() / max(int(critical.sum()), 1))
    return out


class LearnedGate:
    """Shallow tree on probability signals that estimates P(automated decision is correct).

    Its splits become rules of the DMN gate table (e.g. relief_holistic__conf >= 0.87).
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 40, seed: int = 0):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)

    def fit(self, X: np.ndarray, correct, critical=None) -> "LearnedGate":
        """With `critical`, critical-class cases get balanced weight so that the gate's leaves separate
        the cases whose automated miss would be costly, not only the frequent ones."""
        w = None
        if critical is not None:
            critical = np.asarray(critical, bool)
            n_c = max(int(critical.sum()), 1)
            w = np.where(critical, (len(critical) - n_c) / n_c, 1.0)
        self.tree.fit(X, np.asarray(correct, bool), sample_weight=w)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        proba = self.tree.predict_proba(X)
        classes = list(self.tree.classes_)
        return proba[:, classes.index(True)] if True in classes else np.zeros(len(X))
