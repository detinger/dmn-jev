"""Run the typed extractor over the CFPB sample and store one JSON line per case.

    python scripts/extract.py --extractor jev            # real Jev calls (needs TYPESAFE_API_KEY)
    python scripts/extract.py --extractor simulated      # offline development only
    python scripts/extract.py --extractor jev --limit 20 # smoke test
    python scripts/extract.py --extractor jev --pilot 500 # schema pilot on pool cases only
    python scripts/extract.py --extractor jev --reverse-options --subset 1000  # option-order robustness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevdmn.data import cfpb, emscad  # noqa: E402
from jevdmn.schema import load_schema  # noqa: E402


def reversed_options(questions: dict) -> dict:
    out = {}
    for qid, q in questions.items():
        q = dict(q)
        if q["type"] == "choice":
            q["criteria"] = dict(reversed(list(q["criteria"].items())))
        out[qid] = q
    return out


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["jev", "simulated"], required=True)
    ap.add_argument("--config", default="configs/cfpb.yaml")
    ap.add_argument("--sample", default=None, help="default: data/interim/<schema name>_sample.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse-options", action="store_true")
    ap.add_argument("--subset", type=int, default=None, help="random subset of test cases (robustness runs)")
    ap.add_argument("--pilot", type=int, default=None, help="random subset of pool cases only (schema pilot)")
    ap.add_argument("--pilot-class", default=None,
                    help="restrict the pilot to pool cases of this ground-truth class (rare-class check)")
    args = ap.parse_args()

    schema = load_schema(args.config)
    data = {"cfpb": cfpb, "emscad": emscad}[schema.name]
    id_col = getattr(data, "CASE_ID", "complaint_id")
    sample = pd.read_parquet(args.sample or f"data/interim/{schema.name}_sample.parquet")
    splits = [c for c in sample.columns if c.startswith("split_")]
    if args.pilot:
        # Pool only: the pilot must never see calibration or test cases of any task.
        pool = sample[sample[splits].isin(["pool"]).any(axis=1)
                      & (sample[splits].isna() | (sample[splits] == "pool")).all(axis=1)]
        if args.pilot_class:
            gt = {t.ground_truth for t in schema.tasks.values()}
            pool = pool[pool[list(gt)].eq(args.pilot_class).any(axis=1)]
        sample = pool.sample(n=min(args.pilot, len(pool)), random_state=11)
    elif args.subset:
        test = sample[(sample[splits] == "test").any(axis=1)]
        sample = test.sample(n=min(args.subset, len(test)), random_state=7)
    if args.limit:
        sample = sample.head(args.limit)
    questions = schema.payload()
    suffix = ""
    if args.reverse_options:
        questions = reversed_options(questions)
        suffix = "_reversed"

    cases = [(str(r[id_col]), data.state_for(r, schema.max_words)) for r in sample.to_dict("records")]
    if args.extractor == "jev":
        from jevdmn.extractors.jev import JevExtractor
        # request_key sorts keys, so option order does not change it: reversed runs need their own cache.
        extractor = JevExtractor(model=schema.model,
                                 cache_dir="cache/jev_reversed" if args.reverse_options else "cache/jev")
    else:
        from jevdmn.extractors.simulated import SimulatedExtractor
        records = {str(r[id_col]): r for r in sample.to_dict("records")}
        extractor = SimulatedExtractor(records)

    results = extractor.extract(cases, questions)
    if args.limit:
        suffix += "_smoke"  # never overwrite the full run with a smoke test
    elif args.pilot:
        suffix += f"_pilot_{args.pilot_class}" if args.pilot_class else "_pilot"
    elif args.subset and not args.reverse_options:
        suffix = "_subset"
    # CFPB keeps the original file names; other datasets get their name as a prefix.
    prefix = "" if schema.name == "cfpb" else f"{schema.name}_"
    out = Path(f"data/interim/responses_{prefix}{args.extractor}{suffix}.jsonl")
    with out.open("w") as fh:
        for e in results:
            fh.write(json.dumps({"case_id": e.case_id, "extractor": e.extractor, "model": e.response.get("model"),
                                 "latency_ms": e.latency_ms, "timestamp": e.timestamp, "cached": e.cached,
                                 "usage": e.response.get("usage"), "response": e.response}) + "\n")
    models = pd.Series([e.response.get("model") for e in results]).value_counts().to_dict()
    print(f"wrote {len(results)} responses to {out}; models={models}; "
          f"cached={sum(e.cached for e in results)}")


if __name__ == "__main__":
    main()
