"""Run RQ1-RQ4 for every decision task and export results and DMN models.

    python scripts/run_experiments.py --extractor jev
    python scripts/run_experiments.py --extractor simulated   # pipeline check only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevdmn.bpmn import write_process  # noqa: E402
from jevdmn.dmn.reader import read_tables  # noqa: E402
from jevdmn.dmn.writer import DecisionNode, InputNode, SourceNode, write_dmn  # noqa: E402
from jevdmn.experiment import Config, prepare, run_task  # noqa: E402
from jevdmn.schema import load_schema  # noqa: E402


def load_responses(path: Path) -> tuple[dict, pd.DataFrame, set]:
    responses, usage, models = {}, [], set()
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            responses[r["case_id"]] = r["response"]
            usage.append({"case_id": r["case_id"], "latency_ms": r["latency_ms"],
                          "input_tokens": (r.get("usage") or {}).get("input_tokens") or np.nan})
            models.add(r["model"])
    return responses, pd.DataFrame(usage).set_index("case_id"), models


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.DataFrame):
        return None
    raise TypeError(type(o))


def export_drd(res, td, schema, out: Path, model_names: set, n_pool: int, cfg: Config) -> Path:
    business, gate, outcome = res.artefacts["business"], res.artefacts["gate"], res.artefacts["outcome"]
    hq = td.task.holistic.question
    inputs = []
    for s in business.inputs:
        q = schema.question(s.name)
        allowed = ",".join(f'"{v}"' for v in s.domain.values) if s.domain.kind == "enum" else None
        inputs.append(InputNode(s.name, q.label, s.type_ref, allowed, f"Jev typed answer to: {q.instructions}"))
    for s in gate.inputs:
        inputs.append(InputNode(s.name, s.label, "number", None, "Probability signal derived from the Jev answer."))
    inputs.append(InputNode(hq.id, f"Jev decision ({td.task.label})", "string",
                            ",".join(f'"{c}"' for c in td.task.classes), f"Jev holistic answer to: {hq.instructions}"))
    model = ", ".join(sorted(model_names))
    sources = [
        SourceNode("ks_teacher", f"Jev ({model}) holistic decisions",
                   f"Rules distilled from Jev decisions on {n_pool} unlabelled cases (decision tree, merged rules).",
                   authority_for=[business.name]),
        SourceNode("ks_calibration", "Labelled calibration set",
                   f"Gate learned on half of the calibration set; threshold by Learn-then-Test at eps={cfg.eps_primary}, delta={cfg.delta}.",
                   authority_for=[gate.name]),
    ]
    decisions = [
        DecisionNode(business, requires_inputs=[s.name for s in business.inputs],
                     description=f"Distilled decision table ({td.task.description})"),
        DecisionNode(gate, requires_inputs=[s.name for s in gate.inputs],
                     description="Data-derived confidence gate"),
        DecisionNode(outcome, requires_inputs=[hq.id], requires_decisions=[business.name, gate.name],
                     description="Automate only if the gate accepts and the table agrees with Jev"),
    ]
    return write_dmn(out, f"drd_{td.task.name}", f"{td.task.label} (Jev + DMN)", inputs, decisions, sources)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["jev", "simulated"], required=True)
    ap.add_argument("--config", default="configs/cfpb.yaml")
    ap.add_argument("--sample", default="data/interim/cfpb_sample.parquet")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", help="smaller grids for a fast pipeline check")
    args = ap.parse_args()

    schema = load_schema(args.config)
    sample = pd.read_parquet(args.sample)
    responses, usage, models = load_responses(Path(f"data/interim/responses_{args.extractor}.jsonl"))
    cfg = Config()
    if args.quick:
        cfg = Config(depths=(2, 4, 6, None), min_leaf=(20, 100), label_budgets=(250, 1000, None),
                     budget_repeats=2, resamples=50)

    root = Path("results") / args.extractor
    for name in args.tasks or list(schema.tasks):
        task = schema.tasks[name]
        td = prepare(schema, task, sample, responses, usage)
        res = run_task(td, schema, cfg)
        out = root / name
        out.mkdir(parents=True, exist_ok=True)

        res.distillation["sweep"].to_csv(out / "sweep.csv", index=False)
        res.distillation["curve"].to_csv(out / "complexity_curve.csv", index=False)
        res.gate["table"].to_csv(out / "gates.csv", index=False)
        for g, c in res.gate["curves"].items():
            c.to_csv(out / f"risk_coverage_{g}.csv", index=False)
        res.calibration["reliability_raw"].to_csv(out / "reliability_raw.csv", index=False)
        res.calibration["reliability_iso"].to_csv(out / "reliability_iso.csv", index=False)
        res.baselines["learning_curves"].to_csv(out / "learning_curves.csv", index=False)
        for key in ("business", "gate", "outcome"):
            t = res.artefacts[key]
            rows = t.feel_rows()
            pd.DataFrame(rows, columns=[s.label for s in t.inputs] + [t.label]).assign(
                support=[r.support for r in t.rules], purity=[r.purity for r in t.rules]
            ).to_csv(out / f"table_{key}.csv", index=False)

        drd = export_drd(res, td, schema, Path("models") / args.extractor / f"{name}.dmn", models,
                         len(td.part("pool")), cfg)
        write_process(Path("models") / args.extractor / f"{name}.bpmn", name, task.label)
        # Round trip: the exported XML must reproduce the in-memory table on every case.
        parsed = read_tables(drd)
        rt = parsed[res.artefacts["business"].name].evaluate(td.typed)
        roundtrip = float(np.mean(rt == res.artefacts["business"].evaluate(td.typed)))

        summary = {
            "task": name, "extractor": args.extractor, "models": sorted(models),
            "n_cases": len(td.frame), "splits": td.split.value_counts().to_dict(),
            "class_balance": td.y.value_counts(normalize=True).round(4).to_dict(),
            "distillation": {k: v for k, v in res.distillation.items() if k not in ("sweep", "curve")},
            "verification": res.verification, "calibration": {k: v for k, v in res.calibration.items() if "reliability" not in k},
            "gate": {"aurc": res.gate["aurc"], "validity": res.gate["validity"],
                     "agreement_rate_test": res.gate["agreement_rate_test"],
                     "accuracy_when_agree": res.gate["accuracy_when_agree"],
                     "accuracy_when_disagree": res.gate["accuracy_when_disagree"]},
            "baselines": {k: v for k, v in res.baselines.items() if k != "learning_curves"},
            "operational": res.operational, "xml_roundtrip_agreement": roundtrip,
            "config": asdict(cfg),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
        d = summary["distillation"]
        print(f"[{name}] rules={d['n_rules']} inputs={d['n_inputs']} fidelity={d['fidelity_test']['agreement']:.3f} "
              f"kappa={d['fidelity_test']['kappa']:.3f} acc(dmn)={d['accuracy_test']['accuracy']:.3f} "
              f"acc(jev)={summary['baselines']['jev_holistic']['accuracy']:.3f} roundtrip={roundtrip:.3f}")
        print(res.gate["table"].query("gate != 'none' and agreement").to_string(index=False))


if __name__ == "__main__":
    main()
