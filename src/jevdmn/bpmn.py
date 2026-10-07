"""BPMN 2.0 process (Camunda 8 / Zeebe extensions) that executes the Jev + DMN pattern.

    start -> [service] Jev typed extraction -> [business rule] distilled decision
          -> [business rule] automate or review -> <gateway>
                automatic -> [service] apply decision -> end
                manual    -> [user] review case -> [service] record label for re-distillation -> end
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
DC = "http://www.omg.org/spec/DD/20100524/DC"
DI = "http://www.omg.org/spec/DD/20100524/DI"
ZEEBE = "http://camunda.org/schema/zeebe/1.0"
for p, ns in {"bpmn": BPMN, "bpmndi": BPMNDI, "dc": DC, "di": DI, "zeebe": ZEEBE}.items():
    ET.register_namespace(p, ns)


def _e(parent, ns, tag, **attrs):
    return ET.SubElement(parent, f"{{{ns}}}{tag}", {k: str(v) for k, v in attrs.items()})


def write_process(path: str | Path, task: str, label: str) -> Path:
    defs = ET.Element(f"{{{BPMN}}}definitions", {"id": f"defs_{task}", "targetNamespace": "https://fipu.unipu.hr/jevdmn",
                                                  "exporter": "jevdmn", "exporterVersion": "0.1.0"})
    proc = _e(defs, BPMN, "process", id=f"process_{task}", name=f"{label} with Jev and DMN", isExecutable="true")

    nodes = [  # id, type, name, (x, y, w, h)
        ("start", "startEvent", "Case received", (160, 202, 36, 36)),
        ("extract", "serviceTask", "Extract typed facts (Jev)", (250, 180, 120, 80)),
        ("decide", "businessRuleTask", f"Decide: {label.lower()} (DMN)", (420, 180, 120, 80)),
        ("route", "businessRuleTask", "Automate or review (DMN)", (590, 180, 120, 80)),
        ("gw", "exclusiveGateway", "Route?", (765, 195, 50, 50)),
        ("apply", "serviceTask", "Apply decision", (870, 80, 120, 80)),
        ("end_auto", "endEvent", "Decided automatically", (1042, 102, 36, 36)),
        ("review", "userTask", "Review case", (870, 280, 120, 80)),
        ("record", "serviceTask", "Record label for re-distillation", (1040, 280, 120, 80)),
        ("end_manual", "endEvent", "Decided by reviewer", (1212, 302, 36, 36)),
    ]
    flows = [("f1", "start", "extract", None), ("f2", "extract", "decide", None), ("f3", "decide", "route", None),
             ("f4", "route", "gw", None), ("f5", "gw", "apply", '=route = "automatic"'),
             ("f6", "gw", "review", '=route = "manual"'), ("f7", "apply", "end_auto", None),
             ("f8", "review", "record", None), ("f9", "record", "end_manual", None)]
    incoming = {n: [f for f in flows if f[2] == n] for n, *_ in nodes}
    outgoing = {n: [f for f in flows if f[1] == n] for n, *_ in nodes}

    for nid, kind, name, _ in nodes:
        attrs = {"id": nid, "name": name}
        if nid == "gw":
            attrs["default"] = "f6"
        el = _e(proc, BPMN, kind, **attrs)
        ext = None
        if kind in ("serviceTask", "businessRuleTask", "userTask"):
            ext = _e(el, BPMN, "extensionElements")
        if nid == "extract":
            _e(ext, ZEEBE, "taskDefinition", type="jev-extract", retries="3")
        elif nid == "decide":
            _e(ext, ZEEBE, "calledDecision", decisionId=f"decision_{task}_decision", resultVariable=f"{task}_decision")
        elif nid == "route":
            _e(ext, ZEEBE, "calledDecision", decisionId=f"decision_{task}_route", resultVariable="route")
        elif nid == "apply":
            _e(ext, ZEEBE, "taskDefinition", type="apply-decision")
        elif nid == "record":
            _e(ext, ZEEBE, "taskDefinition", type="record-label")
        elif nid == "review":
            _e(ext, ZEEBE, "userTask")
        for f in incoming[nid]:
            _e(el, BPMN, "incoming").text = f[0]
        for f in outgoing[nid]:
            _e(el, BPMN, "outgoing").text = f[0]

    for fid, src, dst, cond in flows:
        sf = _e(proc, BPMN, "sequenceFlow", id=fid, sourceRef=src, targetRef=dst)
        if cond and fid != "f6":
            ce = ET.SubElement(sf, f"{{{BPMN}}}conditionExpression", {"{http://www.w3.org/2001/XMLSchema-instance}type": "bpmn:tFormalExpression"})
            ce.text = cond

    diagram = _e(defs, BPMNDI, "BPMNDiagram", id=f"diagram_{task}")
    plane = _e(diagram, BPMNDI, "BPMNPlane", id=f"plane_{task}", bpmnElement=f"process_{task}")
    box = {}
    for nid, kind, _, (x, y, w, h) in nodes:
        shape = _e(plane, BPMNDI, "BPMNShape", id=f"shape_{nid}", bpmnElement=nid)
        if kind == "exclusiveGateway":
            shape.set("isMarkerVisible", "true")
        _e(shape, DC, "Bounds", x=x, y=y, width=w, height=h)
        box[nid] = (x, y, w, h)
    for fid, src, dst, _ in flows:
        (sx, sy, sw, sh), (tx, ty, tw, th) = box[src], box[dst]
        edge = _e(plane, BPMNDI, "BPMNEdge", id=f"edge_{fid}", bpmnElement=fid)
        start = (sx + sw, sy + sh // 2)
        end = (tx, ty + th // 2)
        pts = [start, end]
        if src == "gw":  # leave the gateway from its top or bottom corner
            gx, gy = sx + sw // 2, (sy if ty < sy else sy + sh)
            pts = [(gx, gy), (gx, ty + th // 2), end]
        for px, py in pts:
            _e(edge, DI, "waypoint", x=px, y=py)

    ET.indent(defs, space="  ")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(defs).write(out, encoding="UTF-8", xml_declaration=True)
    return out
