"""From System One responses to (a) typed DMN inputs, (b) probability signals, (c) tree features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import Question, Schema, Task


def typed_value(q: Question, ans: dict):
    if q.type == "choice":
        return ans["choice"]
    if q.type == "noul":
        return bool(ans["noul"] >= 0.5)
    probs = ans["probabilities"]
    return int(max(probs, key=lambda k: probs[k]))


def typed_frame(responses: dict[str, dict], schema: Schema) -> pd.DataFrame:
    """One row per case, one column per question, holding the DMN-typed answer."""
    rows = {cid: {q.id: typed_value(q, r["answers"][q.id]) for q in schema.questions}
            for cid, r in responses.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    for q in schema.questions:
        if q.type == "noul":
            df[q.id] = df[q.id].astype(bool)
        elif q.type == "score":
            df[q.id] = df[q.id].astype(int)
    return df


def signal_frame(responses: dict[str, dict], schema: Schema) -> pd.DataFrame:
    """Probability-derived signals per question (inputs of the confidence gate)."""
    rows = {}
    for cid, r in responses.items():
        row = {}
        for q in schema.questions:
            a = r["answers"][q.id]
            if q.type == "noul":
                p = float(a["noul"])
                row[f"{q.id}__p"] = p
                row[f"{q.id}__conf"] = abs(2 * p - 1)
            else:
                probs = np.array(sorted(a["probabilities"].values(), reverse=True), dtype=float)
                row[f"{q.id}__p"] = float(probs[0])
                row[f"{q.id}__conf"] = float(a["confidence"])
                row[f"{q.id}__margin"] = float(probs[0] - probs[1]) if len(probs) > 1 else 1.0
        rows[cid] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def holistic_distribution(responses: dict[str, dict], task: Task) -> pd.DataFrame:
    """Jev's probability distribution over the task classes, one row per case."""
    q, h = task.holistic.question, task.holistic
    rows = {}
    for cid, r in responses.items():
        a = r["answers"][q.id]
        if q.type == "noul":
            rows[cid] = {h.true_class: float(a["noul"]), h.false_class: 1 - float(a["noul"])}
        else:
            rows[cid] = {c: float(a["probabilities"].get(c, 0.0)) for c in task.classes}
    df = pd.DataFrame.from_dict(rows, orient="index")[list(task.classes)]
    return df.div(df.sum(axis=1), axis=0)


@dataclass(frozen=True)
class Column:
    """Meaning of one encoded column, used to turn tree splits back into DMN conditions."""
    name: str
    input: str  # question id
    kind: str  # "onehot" | "bool" | "ordinal" | "real"
    value: object = None  # option for one-hot columns


@dataclass
class Encoding:
    columns: list[Column]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def transform(self, typed: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(typed), len(self.columns)), dtype=float)
        for j, c in enumerate(self.columns):
            col = typed[c.input]
            if c.kind == "onehot":
                out[:, j] = (col == c.value).to_numpy(dtype=float)
            else:
                out[:, j] = col.to_numpy(dtype=float)
        return out


def encode_facts(schema: Schema, inputs: list[str] | None = None) -> Encoding:
    """Encoding of fact questions for a decision tree (one-hot for choices)."""
    cols = []
    for q in schema.facts:
        if inputs is not None and q.id not in inputs:
            continue
        if q.type == "choice":
            cols += [Column(f"{q.id}={o}", q.id, "onehot", o) for o in q.domain]
        elif q.type == "noul":
            cols.append(Column(q.id, q.id, "bool"))
        else:
            cols.append(Column(q.id, q.id, "ordinal"))
    return Encoding(cols)


def encode_signals(names: list[str]) -> Encoding:
    return Encoding([Column(n, n, "real") for n in names])
