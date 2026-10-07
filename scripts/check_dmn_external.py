"""Cross-check exported DRDs: XSD validation and evaluation by an independent FEEL engine.

1. xmllint validates each .dmn file against the OMG DMN 1.3 XSD (downloaded on first use).
2. tools/dmn-check/evaluate.mjs evaluates every decision table with feelin (bpmn.io);
   its outputs must equal those of our own FEEL reader on all cases of the task.

    python scripts/check_dmn_external.py --extractor jev
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jevdmn.dmn.reader import read_tables  # noqa: E402
from jevdmn.features import holistic_distribution, signal_frame, typed_frame  # noqa: E402
from jevdmn.schema import load_schema  # noqa: E402
from run_experiments import load_responses  # noqa: E402

XSD_DIR = Path("tools/xsd")
XSD = {"DMN13.xsd": "https://www.omg.org/spec/DMN/20191111/DMN13.xsd",
       "DMNDI13.xsd": "https://www.omg.org/spec/DMN/20191111/DMNDI13.xsd",
       "DC.xsd": "https://www.omg.org/spec/DMN/20180521/DC.xsd",
       "DI.xsd": "https://www.omg.org/spec/DMN/20180521/DI.xsd"}


def validate_xsd(path: Path) -> bool:
    XSD_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in XSD.items():
        if not (XSD_DIR / name).exists():
            urllib.request.urlretrieve(url, XSD_DIR / name)
    r = subprocess.run(["xmllint", "--noout", "--schema", str(XSD_DIR / "DMN13.xsd"), str(path)],
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["jev", "simulated"], required=True)
    ap.add_argument("--config", default="configs/cfpb.yaml")
    args = ap.parse_args()

    schema = load_schema(args.config)
    sample = pd.read_parquet("data/interim/cfpb_sample.parquet")
    responses, _, _ = load_responses(Path(f"data/interim/responses_{args.extractor}.jsonl"))
    report = {}
    for name, task in schema.tasks.items():
        path = Path("models") / args.extractor / f"{name}.dmn"
        ids = sample.loc[sample[f"in_{name}"], "complaint_id"].astype(str).tolist()
        resp = {i: responses[i] for i in ids}
        typed, signals = typed_frame(resp, schema), signal_frame(resp, schema)
        holistic = holistic_distribution(resp, task).idxmax(axis=1)
        tables = read_tables(path)
        business, gate, outcome = (tables[f"{name}_{k}"] for k in ("decision", "gate", "route"))

        data = pd.concat([typed.drop(columns=[task.holistic.question.id]), signals], axis=1)
        data[task.holistic.question.id] = holistic
        ours = pd.DataFrame(index=data.index)
        ours[business.output] = business.evaluate(data)
        ours[gate.output] = gate.evaluate(data)
        derived = pd.DataFrame({gate.output: ours[gate.output],
                                outcome.inputs[1]: ours[business.output] == data[task.holistic.question.id]})
        ours[outcome.output] = outcome.evaluate(derived)

        needed = list(dict.fromkeys(business.inputs + gate.inputs + [task.holistic.question.id]))
        cases = [{k: (bool(v) if isinstance(v, (bool, np.bool_)) else int(v) if isinstance(v, np.integer)
                      else float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}
                 for row in data[needed].to_dict("records")]
        tmp = Path("data/interim") / f"dmn_cases_{args.extractor}_{name}.json"
        tmp.write_text(json.dumps(cases))
        r = subprocess.run(["node", "tools/dmn-check/evaluate.mjs", str(path), str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr)
        theirs = pd.DataFrame(json.loads(r.stdout), index=data.index)
        agree = {col: float((theirs[col].to_numpy() == ours[col].to_numpy()).mean()) for col in ours.columns}
        report[name] = {"xsd_valid": validate_xsd(path), "n_cases": len(data), "agreement": agree}
        print(name, report[name])
    out = Path("results") / args.extractor / "dmn_external_check.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
