"""Read decision tables back from DMN XML and evaluate them from their FEEL text.

This deliberately does not reuse the in-memory rule objects: it parses the
unary tests written to the file, so agreement between the fitted tree and the
evaluated XML checks the exported artefact itself. Supported FEEL subset:
"-", string/number/boolean literals, comparisons (<, <=, >, >=), ranges with
open or closed ends, not(...), and comma-separated disjunctions.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NS = {"dmn": "https://www.omg.org/spec/DMN/20191111/MODEL/"}
_RANGE = re.compile(r"^([\[\(\]])\s*(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)\s*([\]\)\[])$")
_CMP = re.compile(r"^(<=|>=|<|>)\s*(-?[\d.]+)$")


def _split_top(text: str) -> list[str]:
    parts, depth, cur, quoted = [], 0, "", False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if not quoted:
            if ch in "([":
                depth += 1
            elif ch in ")]" and depth:
                depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def unary_test(text: str):
    """Compile a FEEL unary test into a vectorised predicate over a numpy array."""
    text = text.strip()
    if text in ("-", ""):
        return lambda x: np.ones(len(x), dtype=bool)
    if text.startswith("not(") and text.endswith(")"):
        inner = unary_test(text[4:-1])
        return lambda x: ~inner(x)
    parts = _split_top(text)
    if len(parts) > 1:
        tests = [unary_test(p) for p in parts]
        return lambda x: np.logical_or.reduce([t(x) for t in tests])
    t = parts[0]
    if t.startswith('"') and t.endswith('"'):
        v = t[1:-1]
        return lambda x: np.array([a == v for a in x], dtype=bool)
    if t in ("true", "false"):
        v = t == "true"
        return lambda x: np.array([bool(a) == v for a in x], dtype=bool)
    m = _RANGE.match(t)
    if m:
        lo_c, lo, hi, hi_c = m.group(1) == "[", float(m.group(2)), float(m.group(3)), m.group(4) == "]"
        return lambda x: ((x.astype(float) >= lo) if lo_c else (x.astype(float) > lo)) & \
                         ((x.astype(float) <= hi) if hi_c else (x.astype(float) < hi))
    m = _CMP.match(t)
    if m:
        op, v = m.group(1), float(m.group(2))
        ops = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}
        return lambda x: ops[op](x.astype(float), v)
    v = float(t)
    return lambda x: x.astype(float) == v


@dataclass
class ParsedTable:
    decision_id: str
    output: str
    hit_policy: str
    inputs: list[str]  # input expressions
    rules: list[tuple[list[str], str]]  # (input entries, output literal)

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        n = len(data)
        cols = {e: data[e].to_numpy(dtype=object) for e in self.inputs}
        match = np.ones((n, len(self.rules)), dtype=bool)
        for j, (entries, _) in enumerate(self.rules):
            for expr, entry in zip(self.inputs, entries):
                match[:, j] &= unary_test(entry)(cols[expr])
        hits = match.sum(axis=1)
        if self.hit_policy == "UNIQUE" and (hits > 1).any():
            raise ValueError(f"{self.decision_id}: UNIQUE violated on {(hits > 1).sum()} rows")
        outs = np.array([o.strip('"') for _, o in self.rules], dtype=object)
        res = np.full(n, None, dtype=object)
        first = match.argmax(axis=1)
        res[hits > 0] = outs[first[hits > 0]]
        return res


def read_tables(path: str | Path) -> dict[str, ParsedTable]:
    root = ET.parse(path).getroot()
    out = {}
    for dec in root.findall("dmn:decision", NS):
        dt = dec.find("dmn:decisionTable", NS)
        if dt is None:
            continue
        inputs = [i.find("dmn:inputExpression/dmn:text", NS).text for i in dt.findall("dmn:input", NS)]
        output = dt.find("dmn:output", NS).get("name")
        rules = []
        for r in dt.findall("dmn:rule", NS):
            entries = [e.find("dmn:text", NS).text or "" for e in r.findall("dmn:inputEntry", NS)]
            rules.append((entries, r.find("dmn:outputEntry/dmn:text", NS).text))
        out[output] = ParsedTable(dec.get("id"), output, dt.get("hitPolicy", "UNIQUE"), inputs, rules)
    return out
