"""Static analysis and simplification of decision tables.

* overlaps: pairs of rules whose boxes intersect (violations under UNIQUE);
* gaps: regions of the input space covered by no rule (box subtraction);
* merge: rules with the same output that differ in a single input are merged
  when the union is expressible as one condition (Calvanese et al., 2018);
* default rule: an ordered FIRST table that replaces the rules of one output
  by a final catch-all rule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

from .table import DecisionTable, Domain, Rule, _normalise, c_and, c_empty, c_full, c_minus, c_or


def _box(table: DecisionTable, r: Rule) -> list:
    return [r.cond(s) for s in table.inputs]


def _intersects(table: DecisionTable, a: list, b: list) -> bool:
    return all(not c_empty(s.domain, c_and(s.domain, x, y)) for s, x, y in zip(table.inputs, a, b))


def overlaps(table: DecisionTable) -> list[tuple[int, int, bool]]:
    """(i, j, same_output) for every overlapping pair of rules."""
    boxes = [_box(table, r) for r in table.rules]
    return [(i, j, table.rules[i].output == table.rules[j].output)
            for i, j in combinations(range(len(boxes)), 2) if _intersects(table, boxes[i], boxes[j])]


def _subtract(table: DecisionTable, a: list, b: list) -> list[list]:
    if not _intersects(table, a, b):
        return [a]
    pieces, prefix = [], []
    for k, s in enumerate(table.inputs):
        rest = c_minus(s.domain, a[k], b[k])
        if not c_empty(s.domain, rest):
            pieces.append(prefix + [rest] + a[k + 1:])
        prefix.append(c_and(s.domain, a[k], b[k]))
    return pieces


def gaps(table: DecisionTable) -> list[list]:
    """Disjoint boxes of the input space not covered by any rule."""
    uncovered = [[s.domain.full() for s in table.inputs]]
    for r in table.rules:
        b = _box(table, r)
        uncovered = [p for box in uncovered for p in _subtract(table, box, b)]
        if not uncovered:
            break
    return uncovered


def gap_measure(table: DecisionTable, boxes: list[list]) -> float:
    """Fraction of the input space (uniform product measure) left uncovered."""
    total = 0.0
    for box in boxes:
        m = 1.0
        for s, c in zip(table.inputs, box):
            m *= s.domain.measure(c)
        total += m
    return total


def _mergeable(d: Domain, a, b) -> bool:
    if d.finite:
        return True
    return len(_normalise(a + b)) == 1  # adjacent intervals only, to keep cells readable


def merge_rules(table: DecisionTable) -> DecisionTable:
    """Merge same-output rules that differ in exactly one input, until a fixpoint.

    Rules are grouped by (output, conditions on all other inputs); within a group
    finite conditions are united at once and real intervals are chained when adjacent.
    Because the merged rules are disjoint boxes that differ in one input only, the
    union is again a box and the table semantics are unchanged.
    """
    rules = list(table.rules)
    inputs = table.inputs
    changed = True
    while changed:
        changed = False
        for k, s in enumerate(inputs):
            groups: dict[tuple, list[Rule]] = {}
            for r in rules:
                key = (r.output,) + tuple(r.cond(o) for j, o in enumerate(inputs) if j != k)
                groups.setdefault(key, []).append(r)
            merged_rules = []
            for group in groups.values():
                if len(group) == 1:
                    merged_rules.extend(group)
                    continue
                merged = _merge_group(s, group)
                changed |= len(merged) < len(group)
                merged_rules.extend(merged)
            rules = merged_rules
    rules.sort(key=lambda r: (table.outputs.index(r.output) if r.output in table.outputs else 0, -r.support))
    return replace(table, rules=rules)


def _combine(s, rules: list[Rule], cond) -> Rule:
    n = sum(r.support for r in rules)
    purity = sum(r.purity * r.support for r in rules) / n if n else rules[0].purity
    conds = dict(rules[0].conditions)
    if c_full(s.domain, cond):
        conds.pop(s.name, None)
    else:
        conds[s.name] = cond
    return Rule(conds, rules[0].output, n, purity)


def _merge_group(s, group: list[Rule]) -> list[Rule]:
    d = s.domain
    if d.finite:
        cond = frozenset()
        for r in group:
            cond = c_or(d, cond, r.cond(s))
        return [_combine(s, group, cond)]
    # Real input: chain rules whose intervals touch, keeping each cell a single interval.
    items = sorted(group, key=lambda r: (r.cond(s)[0].lo, not r.cond(s)[0].lo_closed))
    out, run, cond = [], [items[0]], items[0].cond(s)
    for r in items[1:]:
        if len(cond) == 1 and len(r.cond(s)) == 1 and _mergeable(d, cond, r.cond(s)):
            run.append(r)
            cond = c_or(d, cond, r.cond(s))
        else:
            out.append(_combine(s, run, cond) if len(run) > 1 else run[0])
            run, cond = [r], r.cond(s)
    out.append(_combine(s, run, cond) if len(run) > 1 else run[0])
    return out


def with_default_rule(table: DecisionTable, output: str | None = None) -> DecisionTable:
    """FIRST-hit table: rules of `output` (default: the most supported) replaced by a final catch-all."""
    if output is None:
        support: dict[str, int] = {}
        for r in table.rules:
            support[r.output] = support.get(r.output, 0) + r.support
        output = max(support, key=support.get)
    kept = [r for r in table.rules if r.output != output]
    n_default = sum(r.support for r in table.rules if r.output == output)
    return replace(table, rules=kept + [Rule({}, output, n_default)], hit_policy="FIRST")


@dataclass
class Report:
    n_rules: int
    n_inputs: int
    mean_conditions: float
    overlaps_conflicting: int
    overlaps_redundant: int
    gap_boxes: int
    gap_fraction: float


def analyse(table: DecisionTable) -> Report:
    ov = overlaps(table)
    if table.hit_policy == "FIRST":
        ov = []  # overlaps are intended under FIRST; order resolves them
    g = gaps(table)
    return Report(table.n_rules(), len(table.used_inputs()), table.mean_conditions(),
                  sum(not same for *_, same in ov), sum(same for *_, same in ov),
                  len(g), gap_measure(table, g))
