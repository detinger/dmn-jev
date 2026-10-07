"""DMN 1.3 XML export of a decision requirements diagram (DRD) with decision tables and DMNDI layout."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .table import DecisionTable, to_feel

DMN = "https://www.omg.org/spec/DMN/20191111/MODEL/"
DMNDI = "https://www.omg.org/spec/DMN/20191111/DMNDI/"
DC = "http://www.omg.org/spec/DMN/20180521/DC/"
DI = "http://www.omg.org/spec/DMN/20180521/DI/"
ET.register_namespace("", DMN)
ET.register_namespace("dmndi", DMNDI)
ET.register_namespace("dc", DC)
ET.register_namespace("di", DI)

W, H = 180, 80  # shape size
GAP_X, GAP_Y = 30, 120


def _id(prefix: str, name: str) -> str:
    return prefix + "_" + re.sub(r"[^A-Za-z0-9_]", "_", name)


def _el(parent, tag, ns=DMN, text=None, **attrs):
    e = ET.SubElement(parent, f"{{{ns}}}{tag}", {k: str(v) for k, v in attrs.items() if v is not None})
    if text is not None:
        e.text = text
    return e


@dataclass
class InputNode:
    name: str
    label: str
    type_ref: str
    allowed: str | None = None  # FEEL unary tests listing allowed values
    description: str = ""


@dataclass
class DecisionNode:
    table: DecisionTable
    requires_inputs: list[str] = field(default_factory=list)
    requires_decisions: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SourceNode:
    id: str
    name: str
    description: str
    authority_for: list[str] = field(default_factory=list)  # decision names


def _levels(decisions: list[DecisionNode]) -> dict[str, int]:
    by_name = {d.table.name: d for d in decisions}
    memo: dict[str, int] = {}

    def level(n):
        if n not in memo:
            deps = by_name[n].requires_decisions
            memo[n] = 1 + max((level(x) for x in deps), default=0)
        return memo[n]

    for d in decisions:
        level(d.table.name)
    return memo


def write_dmn(path: str | Path, model_id: str, model_name: str, inputs: list[InputNode],
              decisions: list[DecisionNode], sources: list[SourceNode] = (), namespace: str = "https://fipu.unipu.hr/jevdmn") -> Path:
    root = ET.Element(f"{{{DMN}}}definitions", {"id": model_id, "name": model_name, "namespace": namespace,
                                                 "exporter": "jevdmn", "exporterVersion": "0.1.0"})
    edges: list[tuple[str, str, str]] = []  # (requirement id, source element id, target element id)

    for inp in inputs:
        if inp.allowed:
            it = _el(root, "itemDefinition", id=_id("type", inp.name), name=_id("t", inp.name))
            _el(it, "typeRef", text=inp.type_ref)
            av = _el(it, "allowedValues", id=_id("allowed", inp.name))
            _el(av, "text", text=inp.allowed)

    for d in decisions:
        t = d.table
        dec = _el(root, "decision", id=_id("decision", t.name), name=t.label)
        if d.description:
            _el(dec, "description", text=d.description)
        _el(dec, "variable", id=_id("var", t.name), name=t.name, typeRef="string")
        for n in d.requires_decisions:
            rid = _id("ir", f"{n}_to_{t.name}")
            ir = _el(dec, "informationRequirement", id=rid)
            _el(ir, "requiredDecision", href="#" + _id("decision", n))
            edges.append((rid, _id("decision", n), _id("decision", t.name)))
        for n in d.requires_inputs:
            rid = _id("ir", f"{n}_to_{t.name}")
            ir = _el(dec, "informationRequirement", id=rid)
            _el(ir, "requiredInput", href="#" + _id("input", n))
            edges.append((rid, _id("input", n), _id("decision", t.name)))
        for s in sources:
            if t.name in s.authority_for:
                rid = _id("ar", f"{s.id}_to_{t.name}")
                ar = _el(dec, "authorityRequirement", id=rid)
                _el(ar, "requiredAuthority", href="#" + s.id)
                edges.append((rid, s.id, _id("decision", t.name)))
        _table_xml(dec, t)

    for inp in inputs:
        e = _el(root, "inputData", id=_id("input", inp.name), name=inp.label)
        if inp.description:
            _el(e, "description", text=inp.description)
        _el(e, "variable", id=_id("var", inp.name), name=inp.name,
            typeRef=_id("t", inp.name) if inp.allowed else inp.type_ref)

    for s in sources:
        ks = _el(root, "knowledgeSource", id=s.id, name=s.name)
        _el(ks, "description", text=s.description)

    _diagram(root, inputs, decisions, sources, edges)
    ET.indent(root, space="  ")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    return out


def _table_xml(dec, t: DecisionTable) -> None:
    dt = _el(dec, "decisionTable", id=_id("table", t.name), hitPolicy=t.hit_policy)
    for k, s in enumerate(t.inputs):
        i = _el(dt, "input", id=_id("in", f"{t.name}_{k}"), label=s.label)
        ie = _el(i, "inputExpression", id=_id("expr", f"{t.name}_{k}"), typeRef=s.type_ref)
        _el(ie, "text", text=s.name)
        if s.domain.kind == "enum":
            iv = _el(i, "inputValues", id=_id("iv", f"{t.name}_{k}"))
            _el(iv, "text", text=",".join(f'"{v}"' for v in s.domain.values))
    o = _el(dt, "output", id=_id("out", t.name), name=t.name, label=t.label, typeRef="string")
    ov = _el(o, "outputValues", id=_id("ov", t.name))
    _el(ov, "text", text=",".join(f'"{v}"' for v in t.outputs))
    _el(dt, "annotation", name="Support")
    _el(dt, "annotation", name="Purity")
    for j, r in enumerate(t.rules):
        rule = _el(dt, "rule", id=_id("rule", f"{t.name}_{j + 1}"))
        for k, s in enumerate(t.inputs):
            ie = _el(rule, "inputEntry", id=_id("ie", f"{t.name}_{j + 1}_{k}"))
            _el(ie, "text", text=to_feel(s.domain, r.cond(s)))
        oe = _el(rule, "outputEntry", id=_id("oe", f"{t.name}_{j + 1}"))
        _el(oe, "text", text=f'"{r.output}"')
        a = _el(rule, "annotationEntry")
        _el(a, "text", text=str(r.support))
        a = _el(rule, "annotationEntry")
        _el(a, "text", text="" if r.purity != r.purity else f"{r.purity:.3f}")


def _diagram(root, inputs, decisions, sources, edges) -> None:
    levels = _levels(decisions)
    top = max(levels.values(), default=1)
    pos: dict[str, tuple[int, int]] = {}

    def row(ids: list[str], y: int, width_total: int):
        x0 = max(0, (width_total - len(ids) * (W + GAP_X)) // 2)
        for k, i in enumerate(ids):
            pos[i] = (x0 + k * (W + GAP_X), y)

    n_inputs = len(inputs)
    width = max(n_inputs, 1) * (W + GAP_X)
    for lv in range(top, 0, -1):
        ids = [_id("decision", d.table.name) for d in decisions if levels[d.table.name] == lv]
        row(ids, (top - lv) * (H + GAP_Y), width)
    row([_id("input", i.name) for i in inputs], top * (H + GAP_Y), width)
    # Knowledge sources sit to the right of the decision they govern.
    right = max((x for x, _ in pos.values()), default=0) + W + 2 * GAP_X
    for k, s in enumerate(sources):
        target = _id("decision", s.authority_for[0]) if s.authority_for else None
        y = pos[target][1] if target in pos else k * (H + GAP_Y)
        pos[s.id] = (right + k * (W + GAP_X), y)

    dmndi = _el(root, "DMNDI", ns=DMNDI)
    diag = _el(dmndi, "DMNDiagram", ns=DMNDI, id="DMNDiagram_1")
    for eid, (x, y) in pos.items():
        sh = _el(diag, "DMNShape", ns=DMNDI, id=f"shape_{eid}", dmnElementRef=eid)
        _el(sh, "Bounds", ns=DC, x=x, y=y, width=W, height=H)
    for rid, src, dst in edges:
        (sx, sy), (dx, dy) = pos[src], pos[dst]
        e = _el(diag, "DMNEdge", ns=DMNDI, id=f"edge_{rid}", dmnElementRef=rid)
        if sy == dy:  # same row (knowledge source beside its decision)
            left_to_right = sx < dx
            _el(e, "waypoint", ns=DI, x=sx + (W if left_to_right else 0), y=sy + H // 2)
            _el(e, "waypoint", ns=DI, x=dx + (0 if left_to_right else W), y=dy + H // 2)
        elif sy > dy:  # source below target: leave from the top edge, enter at the bottom edge
            _el(e, "waypoint", ns=DI, x=sx + W // 2, y=sy)
            _el(e, "waypoint", ns=DI, x=dx + W // 2, y=dy + H)
        else:
            _el(e, "waypoint", ns=DI, x=sx + W // 2, y=sy + H)
            _el(e, "waypoint", ns=DI, x=dx + W // 2, y=dy)
