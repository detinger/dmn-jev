"""Common interface for typed extractors (Jev, LLM via adapter, simulator).

Every extractor returns the System One response format, so the rest of the
pipeline does not depend on which model produced the answers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


@dataclass
class Extraction:
    case_id: str
    response: dict  # System One response: {"model", "answers", "usage"}
    latency_ms: float
    timestamp: str
    extractor: str
    cached: bool = False
    meta: dict = field(default_factory=dict)


class Extractor(Protocol):
    name: str

    def extract(self, cases: Iterable[tuple[str, dict]], questions: dict) -> list[Extraction]: ...


def request_key(model: str, state, questions: dict) -> str:
    # sort_keys makes the key blind to option order; option-order runs use a separate cache directory.
    blob = json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class DiskCache:
    """One JSON file per request, holding the request and the raw response.

    The cache is the experimental record: re-running the pipeline never
    re-queries a request that already has a stored response.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, record: dict) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(p)
