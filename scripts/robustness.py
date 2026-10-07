"""Option-order robustness of Choice answers (jev-1.13 is known to favour the first option).

After the full run, query a random subset of test cases again with reversed Choice options
and compare with the original answers of the same cases:
    python scripts/extract.py --extractor jev --subset 1000 --reverse-options
    python scripts/robustness.py --extractor jev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiments import load_responses  # noqa: E402

from jevdmn.metrics import agreement  # noqa: E402
from jevdmn.schema import load_schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="jev")
    ap.add_argument("--config", default="configs/cfpb.yaml")
    args = ap.parse_args()
    schema = load_schema(args.config)
    base, _, _ = load_responses(Path(f"data/interim/responses_{args.extractor}.jsonl"))
    rev, _, _ = load_responses(Path(f"data/interim/responses_{args.extractor}_reversed.jsonl"))
    ids = sorted(set(base) & set(rev))
    out = {"n": len(ids)}
    for q in schema.questions:
        if q.type != "choice":
            continue
        a = [base[i]["answers"][q.id]["choice"] for i in ids]
        b = [rev[i]["answers"][q.id]["choice"] for i in ids]
        first = list(q.criteria)[0]
        last = list(q.criteria)[-1]
        out[q.id] = {**agreement(a, b),
                     "share_first_option_original": sum(x == first for x in a) / len(a),
                     "share_first_option_reversed_order": sum(x == last for x in b) / len(b)}
    Path("results").joinpath(args.extractor).mkdir(parents=True, exist_ok=True)
    Path(f"results/{args.extractor}/option_order.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
