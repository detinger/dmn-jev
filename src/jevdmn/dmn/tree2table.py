"""Translate a fitted scikit-learn decision tree into a DMN decision table.

Every root-to-leaf path becomes one rule. Splits on encoded columns are mapped
back to conditions on the original typed inputs (one-hot -> enumeration sets,
booleans -> true/false, ordinal -> integer ranges, real -> intervals). Real
thresholds are replaced by the shortest decimal that induces the same
partition of the training data, which keeps the table readable without
changing any training-set decision.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..features import Encoding
from .table import Domain, DecisionTable, InputSpec, Interval, Rule, c_and, c_empty

INF = math.inf


def nice_threshold(a: float, b: float) -> float:
    """Shortest decimal c with a <= c < b (so 'x <= c' splits {a} from {b} as before)."""
    for digits in range(1, 8):
        c = math.ceil(a * 10 ** digits - 1e-9) / 10 ** digits
        if a <= c < b:
            return c
    return a


def tree_to_table(tree: DecisionTreeClassifier, enc: Encoding, specs: list[InputSpec],
                  name: str, label: str, X_train: np.ndarray | None = None,
                  leaf_output=None, outputs: tuple[str, ...] | None = None) -> DecisionTable:
    """`leaf_output(dist, classes) -> (output, score)` overrides the majority-class output of a leaf."""
    t = tree.tree_
    classes = [str(c) for c in tree.classes_]
    spec_by_name = {s.name: s for s in specs}
    rules: list[Rule] = []

    def threshold(node: int) -> float:
        thr = float(t.threshold[node])
        if X_train is None:
            return thr
        col = X_train[:, t.feature[node]]
        lower, upper = col[col <= thr], col[col > thr]
        if len(lower) == 0 or len(upper) == 0:
            return thr
        return nice_threshold(float(lower.max()), float(upper.min()))

    def walk(node: int, conds: dict):
        if t.children_left[node] == -1:
            dist = t.value[node][0] / t.value[node][0].sum()
            if leaf_output is None:
                k = int(np.argmax(dist))
                out, score = classes[k], float(dist[k])
            else:
                out, score = leaf_output(dist, classes)
            rules.append(Rule(dict(conds), out, support=int(t.n_node_samples[node]), purity=score))
            return
        col = enc.columns[t.feature[node]]
        spec = spec_by_name[col.input]
        d = spec.domain
        cur = conds.get(spec.name, d.full())
        if col.kind == "onehot":
            left, right = d.full() - {col.value}, frozenset({col.value})
        elif col.kind == "bool":
            left, right = frozenset({False}), frozenset({True})
        elif col.kind == "ordinal":
            thr = t.threshold[node]
            left = frozenset(v for v in d.values if v <= thr)
            right = frozenset(v for v in d.values if v > thr)
        else:
            thr = threshold(node)
            left = (Interval(-INF, thr, False, True),)
            right = (Interval(thr, INF, False, False),)
        for child, part in ((t.children_left[node], left), (t.children_right[node], right)):
            new = c_and(d, cur, part)
            if c_empty(d, new):
                continue  # unreachable branch given earlier splits
            walk(child, {**conds, spec.name: new})

    walk(0, {})
    return DecisionTable(name=name, label=label, inputs=list(specs), outputs=tuple(outputs or classes), rules=rules)


def fact_specs(schema, inputs: list[str] | None = None) -> list[InputSpec]:
    specs = []
    for q in schema.facts:
        if inputs is not None and q.id not in inputs:
            continue
        kind = {"choice": "enum", "noul": "bool", "score": "ordinal"}[q.type]
        specs.append(InputSpec(q.id, q.label or q.id, Domain(kind, tuple(q.domain))))
    return specs


def signal_specs(names: list[str], labels: dict[str, str] | None = None) -> list[InputSpec]:
    labels = labels or {}
    return [InputSpec(n, labels.get(n, n), Domain("real", lo=0.0, hi=1.0)) for n in names]
