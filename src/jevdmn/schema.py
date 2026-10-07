"""Typed question schema shared by the extractor, the feature encoder and the DMN writer.

A System One question has one of three types. Each maps to a DMN type:

    choice -> string with an enumerated domain (the option keys)
    noul   -> boolean (probability of "yes" thresholded at 0.5)
    score  -> integer level 0..k-1 (most likely level)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Question:
    id: str
    type: str  # "choice" | "noul" | "score"
    instructions: str
    label: str = ""
    criteria: Any = None  # dict for choice, list for score, optional dict for noul

    @property
    def domain(self) -> tuple:
        if self.type == "choice":
            return tuple(self.criteria)
        if self.type == "noul":
            return (False, True)
        return tuple(range(len(self.criteria)))

    @property
    def dmn_type(self) -> str:
        return {"choice": "string", "noul": "boolean", "score": "number"}[self.type]

    def payload(self) -> dict:
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


@dataclass(frozen=True)
class Holistic:
    question: Question
    true_class: str | None = None  # noul only
    false_class: str | None = None

    def to_class(self, typed_value) -> str:
        if self.question.type == "noul":
            return self.true_class if typed_value else self.false_class
        return str(typed_value)


@dataclass(frozen=True)
class Task:
    name: str
    label: str
    description: str
    ground_truth: str
    classes: tuple[str, ...]
    holistic: Holistic
    subset: dict | None = None


@dataclass
class Schema:
    name: str
    model: str
    state_fields: dict
    max_words: int
    facts: list[Question]
    tasks: dict[str, Task] = field(default_factory=dict)

    @property
    def questions(self) -> list[Question]:
        return self.facts + [t.holistic.question for t in self.tasks.values()]

    def question(self, qid: str) -> Question:
        return next(q for q in self.questions if q.id == qid)

    def payload(self) -> dict:
        return {q.id: q.payload() for q in self.questions}


def load_schema(path: str | Path) -> Schema:
    cfg = yaml.safe_load(Path(path).read_text())
    facts = [Question(id=f["id"], type=f["type"], instructions=f["instructions"],
                      label=f.get("label", f["id"]), criteria=f.get("criteria")) for f in cfg["facts"]]
    tasks = {}
    for name, d in cfg["decisions"].items():
        h = d["holistic"]
        q = Question(id=h["id"], type=h["type"], instructions=h["instructions"],
                     label=d["label"], criteria=h.get("criteria"))
        tasks[name] = Task(name=name, label=d["label"], description=d["description"],
                           ground_truth=d["ground_truth"], classes=tuple(d["classes"]),
                           holistic=Holistic(q, h.get("true_class"), h.get("false_class")),
                           subset=d.get("subset"))
    return Schema(name=cfg["name"], model=cfg["model"], state_fields=cfg["state"]["fields"],
                  max_words=cfg["state"]["max_words"], facts=facts, tasks=tasks)
