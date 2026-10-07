"""Decision tables over typed domains: conditions, rules, evaluation and FEEL rendering.

A condition restricts one input. Finite inputs (enum, boolean, ordinal) use the
set of allowed values; real inputs (probabilities) use a union of intervals.
Each rule is therefore a box in the input space, which makes overlap and gap
analysis exact (cf. the geometric view of Calvanese et al., 2018).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
import pandas as pd

INF = math.inf


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    lo_closed: bool
    hi_closed: bool

    def empty(self) -> bool:
        return self.lo > self.hi or (self.lo == self.hi and not (self.lo_closed and self.hi_closed))

    def intersect(self, o: "Interval") -> "Interval":
        if self.lo > o.lo or (self.lo == o.lo and not self.lo_closed):
            lo, lc = self.lo, self.lo_closed
        else:
            lo, lc = o.lo, o.lo_closed
        if self.hi < o.hi or (self.hi == o.hi and not self.hi_closed):
            hi, hc = self.hi, self.hi_closed
        else:
            hi, hc = o.hi, o.hi_closed
        return Interval(lo, hi, lc, hc)

    def contains(self, x: np.ndarray) -> np.ndarray:
        left = (x >= self.lo) if self.lo_closed else (x > self.lo)
        right = (x <= self.hi) if self.hi_closed else (x < self.hi)
        return left & right


def _normalise(ivs: Iterable[Interval]) -> tuple[Interval, ...]:
    ivs = sorted((i for i in ivs if not i.empty()), key=lambda i: (i.lo, not i.lo_closed))
    out: list[Interval] = []
    for i in ivs:
        if out:
            last = out[-1]
            touches = i.lo < last.hi or (i.lo == last.hi and (i.lo_closed or last.hi_closed))
            if touches:
                if i.hi > last.hi or (i.hi == last.hi and i.hi_closed):
                    out[-1] = Interval(last.lo, i.hi, last.lo_closed, i.hi_closed)
                continue
        out.append(i)
    return tuple(out)


@dataclass(frozen=True)
class Domain:
    kind: str  # "enum" | "bool" | "ordinal" | "real"
    values: tuple = ()  # finite domains, in display order
    lo: float = 0.0  # real domains
    hi: float = 1.0

    @property
    def finite(self) -> bool:
        return self.kind != "real"

    def full(self):
        return frozenset(self.values) if self.finite else (Interval(self.lo, self.hi, True, True),)

    def measure(self, cond) -> float:
        if self.finite:
            return len(cond) / len(self.values)
        return sum(i.hi - i.lo for i in cond) / (self.hi - self.lo)


# Condition algebra --------------------------------------------------------

def c_empty(d: Domain, c) -> bool:
    return len(c) == 0


def c_full(d: Domain, c) -> bool:
    if d.finite:
        return c >= frozenset(d.values)
    return _normalise(c) == _normalise(d.full())


def c_and(d: Domain, a, b):
    if d.finite:
        return a & b
    return _normalise(x.intersect(y) for x in a for y in b)


def c_or(d: Domain, a, b):
    return (a | b) if d.finite else _normalise(a + b)


def c_not(d: Domain, a):
    if d.finite:
        return frozenset(d.values) - a
    pieces, lo, lc = [], d.lo, True
    for i in _normalise(a):
        pieces.append(Interval(lo, i.lo, lc, not i.lo_closed))
        lo, lc = i.hi, not i.hi_closed
    pieces.append(Interval(lo, d.hi, lc, True))
    return _normalise(pieces)


def c_minus(d: Domain, a, b):
    return c_and(d, a, c_not(d, b))


def c_contains(d: Domain, c, x: np.ndarray) -> np.ndarray:
    if d.finite:
        return np.isin(x, np.array(sorted(c, key=str), dtype=object)) if len(c) else np.zeros(len(x), bool)
    x = x.astype(float)
    m = np.zeros(len(x), dtype=bool)
    for i in c:
        m |= i.contains(x)
    return m


# FEEL rendering -------------------------------------------------------------

def fmt_num(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def fmt_value(d: Domain, v) -> str:
    if d.kind == "enum":
        return f'"{v}"'
    if d.kind == "bool":
        return "true" if v else "false"
    return fmt_num(v)


def to_feel(d: Domain, c) -> str:
    if c_full(d, c):
        return "-"
    if d.kind == "enum":
        inside = [v for v in d.values if v in c]
        outside = [v for v in d.values if v not in c]
        if len(inside) <= len(outside):
            return ",".join(fmt_value(d, v) for v in inside)
        return "not(" + ",".join(fmt_value(d, v) for v in outside) + ")"
    if d.kind == "bool":
        return "true" if True in c else "false"
    if d.kind == "ordinal":
        vals = [v for v in d.values if v in c]
        runs, start = [], vals[0]
        for a, b in zip(vals, vals[1:] + [None]):
            if b is None or d.values.index(b) != d.values.index(a) + 1:
                runs.append((start, a))
                start = b
        parts = []
        for a, b in runs:
            if a == b:
                parts.append(fmt_num(a))
            elif a == d.values[0]:
                parts.append(f"<= {fmt_num(b)}")
            elif b == d.values[-1]:
                parts.append(f">= {fmt_num(a)}")
            else:
                parts.append(f"[{fmt_num(a)}..{fmt_num(b)}]")
        return ", ".join(parts)
    parts = []
    for i in _normalise(c):
        at_lo, at_hi = i.lo <= d.lo, i.hi >= d.hi
        if at_lo and not at_hi:
            parts.append(f"{'<=' if i.hi_closed else '<'} {fmt_num(i.hi)}")
        elif at_hi and not at_lo:
            parts.append(f"{'>=' if i.lo_closed else '>'} {fmt_num(i.lo)}")
        else:
            parts.append(f"{'[' if i.lo_closed else '('}{fmt_num(i.lo)}..{fmt_num(i.hi)}{']' if i.hi_closed else ')'}")
    return ", ".join(parts)


# Tables -------------------------------------------------------------------

@dataclass(frozen=True)
class InputSpec:
    name: str  # FEEL variable name
    label: str
    domain: Domain

    @property
    def type_ref(self) -> str:
        return {"enum": "string", "bool": "boolean", "ordinal": "number", "real": "number"}[self.domain.kind]


@dataclass
class Rule:
    conditions: dict  # input name -> condition (missing = any value)
    output: str
    support: int = 0
    purity: float = float("nan")

    def cond(self, spec: InputSpec):
        return self.conditions.get(spec.name, spec.domain.full())


@dataclass
class DecisionTable:
    name: str  # FEEL variable name of the output
    label: str
    inputs: list[InputSpec]
    outputs: tuple[str, ...]
    rules: list[Rule]
    hit_policy: str = "UNIQUE"  # or "FIRST"
    meta: dict = field(default_factory=dict)

    # Complexity
    def n_rules(self) -> int:
        return len(self.rules)

    def used_inputs(self) -> list[InputSpec]:
        return [s for s in self.inputs if any(not c_full(s.domain, r.cond(s)) for r in self.rules)]

    def mean_conditions(self) -> float:
        if not self.rules:
            return 0.0
        return float(np.mean([sum(not c_full(s.domain, r.cond(s)) for s in self.inputs) for r in self.rules]))

    def drop_unused_inputs(self) -> "DecisionTable":
        used = {s.name for s in self.used_inputs()}
        rules = [replace(r, conditions={k: v for k, v in r.conditions.items() if k in used}) for r in self.rules]
        return replace(self, inputs=[s for s in self.inputs if s.name in used], rules=rules)

    # Semantics
    def match_matrix(self, data: pd.DataFrame) -> np.ndarray:
        m = np.ones((len(data), len(self.rules)), dtype=bool)
        for j, r in enumerate(self.rules):
            for s in self.inputs:
                c = r.cond(s)
                if not c_full(s.domain, c):
                    m[:, j] &= c_contains(s.domain, c, data[s.name].to_numpy(dtype=object if s.domain.finite else float))
        return m

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        m = self.match_matrix(data)
        out = np.full(len(data), None, dtype=object)
        hits = m.sum(axis=1)
        if self.hit_policy == "UNIQUE":
            if (hits > 1).any():
                raise ValueError(f"UNIQUE hit policy violated on {(hits > 1).sum()} rows")
            idx = m.argmax(axis=1)
        else:
            idx = m.argmax(axis=1)  # first matching rule
        outputs = np.array([r.output for r in self.rules], dtype=object)
        out[hits > 0] = outputs[idx[hits > 0]]
        return out

    def feel_rows(self) -> list[list[str]]:
        return [[to_feel(s.domain, r.cond(s)) for s in self.inputs] + [r.output] for r in self.rules]
