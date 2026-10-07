"""Jev (TypeSafe System One API) extractor with a persistent request cache."""

from __future__ import annotations

import asyncio
import os
import random
import time
from datetime import datetime, timezone
from typing import Iterable

import httpx
from tqdm import tqdm

from .base import DiskCache, Extraction, request_key

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class JevExtractor:
    name = "jev"

    def __init__(self, model: str | None = None, cache_dir: str = "cache/jev",
                 api_key: str | None = None, base_url: str | None = None,
                 concurrency: int = 16, max_retries: int = 8, timeout: float = 60.0):
        self.model = model or os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.url = (base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/") + "/v1/systemone"
        self.cache = DiskCache(cache_dir)
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout

    def extract(self, cases: Iterable[tuple[str, dict]], questions: dict) -> list[Extraction]:
        return asyncio.run(self._extract_all(list(cases), questions))

    async def _extract_all(self, cases, questions) -> list[Extraction]:
        todo, done = [], {}
        for case_id, state in cases:
            key = request_key(self.model, state, questions)
            hit = self.cache.get(key)
            if hit is not None:
                done[case_id] = _from_record(case_id, hit, cached=True)
            else:
                todo.append((case_id, state, key))
        if todo and not self.api_key:
            raise RuntimeError(f"{len(todo)} uncached requests and TYPESAFE_API_KEY is not set")

        sem = asyncio.Semaphore(self.concurrency)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            bar = tqdm(total=len(todo), desc="jev", disable=not todo)

            async def one(case_id, state, key):
                async with sem:
                    rec = await self._call(client, state, questions)
                self.cache.put(key, rec)
                done[case_id] = _from_record(case_id, rec, cached=False)
                bar.update()

            await asyncio.gather(*(one(*t) for t in todo))
            bar.close()
        return [done[cid] for cid, _ in cases]

    async def _call(self, client: httpx.AsyncClient, state, questions) -> dict:
        body = {"model": self.model, "state": state, "questions": questions}
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = await client.post(self.url, json=body)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(_backoff(attempt))
                continue
            latency = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                return {
                    "request": body,
                    "response": r.json(),
                    "latency_ms": latency,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": r.headers.get("x-typesafe-request-id"),
                    "attempts": attempt + 1,
                }
            if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                raise RuntimeError(f"Jev API error {r.status_code}: {r.text[:500]}")
            retry_after = r.headers.get("retry-after")
            await asyncio.sleep(float(retry_after) if retry_after else _backoff(attempt))
        raise RuntimeError("unreachable")


def _backoff(attempt: int) -> float:
    return min(30.0, 0.5 * 2 ** attempt) * (0.5 + random.random())


def _from_record(case_id: str, rec: dict, cached: bool) -> Extraction:
    return Extraction(case_id=case_id, response=rec["response"], latency_ms=rec["latency_ms"],
                      timestamp=rec["timestamp"], extractor="jev", cached=cached,
                      meta={"request_id": rec.get("request_id")})
