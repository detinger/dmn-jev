"""End-to-end experiment for one decision task (RQ1-RQ4)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import calibration as cal
from . import gating, metrics
from .dmn.table import DecisionTable, Domain, InputSpec, Rule
from .dmn.tree2table import fact_specs, signal_specs, tree_to_table
from .dmn.verify import analyse, merge_rules, with_default_rule
from .features import encode_facts, encode_signals, holistic_distribution, signal_frame, typed_frame
from .schema import Schema, Task

PRICE_PER_MTOK = 0.042  # USD per million input tokens, jev-1.13 (output tokens are free)


@dataclass
class Config:
    depths: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 12, None)
    min_leaf: tuple = (5, 20, 50, 100)
    fidelity_tol: float = 0.01
    eps_grid: tuple = (0.01, 0.02, 0.05, 0.10)
    eps_primary: float = 0.05
    delta: float = 0.05
    gate_depth: int = 3
    gate_min_leaf: int = 40
    label_budgets: tuple = (100, 250, 500, 1000, 2000, None)
    budget_repeats: int = 5
    resamples: int = 200
    seed: int = 0


@dataclass
class TaskData:
    task: Task
    frame: pd.DataFrame  # sample rows of this task (index = case id)
    typed: pd.DataFrame
    signals: pd.DataFrame
    jev_dist: pd.DataFrame
    jev_pred: pd.Series
    split: pd.Series
    y: pd.Series
    usage: pd.DataFrame  # latency_ms, input_tokens per case

    def part(self, name: str) -> pd.Index:
        return self.split.index[self.split == name]


def prepare(schema: Schema, task: Task, sample: pd.DataFrame, responses: dict[str, dict],
            usage: pd.DataFrame) -> TaskData:
    frame = sample[sample[f"in_{task.name}"]].copy()
    frame.index = frame["case_id" if "case_id" in frame else "complaint_id"].astype(str)
    resp = {cid: responses[cid] for cid in frame.index}
    typed = typed_frame(resp, schema).loc[frame.index]
    signals = signal_frame(resp, schema).loc[frame.index]
    dist = holistic_distribution(resp, task).loc[frame.index]
    return TaskData(task, frame, typed, signals, dist, dist.idxmax(axis=1), frame[f"split_{task.name}"],
                    frame[task.ground_truth].astype(str), usage.loc[frame.index])


# RQ1: distillation -------------------------------------------------------------------------

def _fit_table(X, y, enc, specs, depth, leaf, name, label, seed) -> tuple[DecisionTreeClassifier, DecisionTable]:
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed).fit(X, y)
    table = merge_rules(tree_to_table(tree, enc, specs, name, label, X_train=X)).drop_unused_inputs()
    return tree, table


def distil(td: TaskData, schema: Schema, cfg: Config, target: str = "jev", ids: pd.Index | None = None) -> dict:
    """Fit a decision table on typed facts. target='jev' distils Jev's holistic decision; 'gt' uses labels."""
    enc = encode_facts(schema)
    specs = fact_specs(schema)
    ids = td.part("pool") if ids is None else ids
    X_all = enc.transform(td.typed)
    pos = {cid: k for k, cid in enumerate(td.typed.index)}
    rows = np.array([pos[c] for c in ids])
    y_src = (td.jev_pred if target == "jev" else td.y).loc[ids].to_numpy()

    rng = np.random.default_rng(cfg.seed)
    inner = rng.permutation(len(rows))
    cut = int(0.8 * len(rows))
    tr, va = rows[inner[:cut]], rows[inner[cut:]]
    ytr, yva = y_src[inner[:cut]], y_src[inner[cut:]]

    name = f"{td.task.name}_decision"
    sweep = []
    for depth in cfg.depths:
        for leaf in cfg.min_leaf:
            _, table = _fit_table(X_all[tr], ytr, enc, specs, depth, leaf, name, td.task.label, cfg.seed)
            val_score = float(np.mean(table.evaluate(td.typed.iloc[va]) == yva))
            sweep.append({"depth": depth if depth is not None else 0, "min_leaf": leaf, "n_rules": table.n_rules(),
                          "n_inputs": len(table.inputs), "val": val_score})
    sweep = pd.DataFrame(sweep)
    best = sweep["val"].max()
    pick = sweep[sweep["val"] >= best - cfg.fidelity_tol].sort_values(["n_rules", "val"], ascending=[True, False]).iloc[0]
    depth = None if int(pick["depth"]) == 0 else int(pick["depth"])

    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=int(pick["min_leaf"]), random_state=cfg.seed)
    tree.fit(X_all[rows], y_src)
    raw = tree_to_table(tree, enc, specs, name, td.task.label, X_train=X_all[rows])
    table = merge_rules(raw).drop_unused_inputs()

    # Test-set fidelity of every configuration, refitted on the full source set (complexity curve).
    test = td.part("test")
    curve = []
    for _, r in sweep.iterrows():
        d = None if int(r["depth"]) == 0 else int(r["depth"])
        _, t = _fit_table(X_all[rows], y_src, enc, specs, d, int(r["min_leaf"]), name, td.task.label, cfg.seed)
        out = t.evaluate(td.typed.loc[test])
        curve.append({**r.to_dict(), "n_rules_full": t.n_rules(), "mean_conditions": t.mean_conditions(),
                      "test_fidelity": float(np.mean(out == td.jev_pred.loc[test].to_numpy())),
                      "test_accuracy": float(np.mean(out == td.y.loc[test].to_numpy()))})
    return {"table": table, "raw_table": raw, "tree": tree, "sweep": sweep, "curve": pd.DataFrame(curve),
            "choice": {"depth": depth, "min_leaf": int(pick["min_leaf"])}}


def table_report(raw: DecisionTable, table: DecisionTable) -> dict:
    a, b = analyse(raw), analyse(table)
    first = with_default_rule(table)
    return {"before_merge": asdict(a), "after_merge": asdict(b),
            "first_hit": {"n_rules": first.n_rules(), "mean_conditions": first.mean_conditions()}}


# RQ3: confidence gate ----------------------------------------------------------------------

def gate_inputs(td: TaskData, table: DecisionTable) -> list[str]:
    hq = td.task.holistic.question.id
    names = [f"{hq}__p", f"{hq}__conf"] + ([f"{hq}__margin"] if f"{hq}__margin" in td.signals else [])
    for s in table.inputs:
        names += [c for c in (f"{s.name}__conf",) if c in td.signals]
    return names


def run_gates(td: TaskData, dmn_out: pd.Series, table: DecisionTable, cfg: Config) -> dict:
    calib, test = td.part("calibration"), td.part("test")
    correct = dmn_out == td.y
    agree = dmn_out == td.jev_pred
    jev_conf = td.jev_dist.max(axis=1)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(calib))
    calib_a, calib_b = calib[perm[: len(calib) // 2]], calib[perm[len(calib) // 2:]]

    names = gate_inputs(td, table)
    learned = gating.LearnedGate(cfg.gate_depth, cfg.gate_min_leaf, cfg.seed)
    learned.fit(td.signals.loc[calib_a, names].to_numpy(), correct.loc[calib_a].to_numpy())
    learned_score = pd.Series(learned.score(td.signals[names].to_numpy()), index=td.signals.index)

    scores = {"none": pd.Series(1.0, index=td.y.index), "jev_confidence": jev_conf, "learned": learned_score}
    rows, thresholds = [], {}
    for gname, score in scores.items():
        for use_agree in (False, True):
            s = score.where(agree, -np.inf) if use_agree else score
            for eps in cfg.eps_grid:
                # Split A fixes the gate and the ordered thresholds; split B tests them.
                thr = gating.calibrate(s.loc[calib_a], correct.loc[calib_a], s.loc[calib_b], correct.loc[calib_b], eps, cfg.delta)
                acc = gating.apply(s.loc[test], thr)
                m = gating.selective_metrics(acc, correct.loc[test])
                rows.append({"gate": gname, "agreement": use_agree, "eps": eps, "threshold": thr.value,
                             "calib_coverage": thr.calib_coverage, **m})
                thresholds[(gname, use_agree, eps)] = thr

    curves = {g: gating.risk_coverage(scores[g].loc[test], correct.loc[test]) for g in ("jev_confidence", "learned")}
    aurc = {g: gating.aurc(scores[g].loc[test], correct.loc[test]) for g in ("jev_confidence", "learned")}
    return {"table": pd.DataFrame(rows), "thresholds": thresholds, "learned": learned, "gate_inputs": names,
            "curves": curves, "aurc": aurc, "scores": scores, "correct": correct, "agree": agree}


def validity_check(score: pd.Series, correct: pd.Series, ids: pd.Index, eps: float, cfg: Config) -> dict:
    """Empirical check of the LTT guarantee: repeated random A/B/test splits of `ids`."""
    rng = np.random.default_rng(cfg.seed + 1)
    viol, cov = [], []
    s, c = score.loc[ids].to_numpy(), correct.loc[ids].to_numpy()
    for _ in range(cfg.resamples):
        p = rng.permutation(len(ids))
        a, b, t = np.array_split(p, [len(p) // 4, len(p) // 2])
        thr = gating.calibrate(s[a], c[a], s[b], c[b], eps, cfg.delta)
        b = t
        acc = s[b] >= thr.value
        risk = float((~c[b][acc]).mean()) if acc.any() else 0.0
        viol.append(risk > eps)
        cov.append(acc.mean())
    return {"eps": eps, "violation_rate": float(np.mean(viol)), "mean_coverage": float(np.mean(cov))}


def gate_table(td: TaskData, gates: dict, cfg: Config) -> DecisionTable:
    """DMN table of the learned gate at the primary risk level (outputs 'auto' / 'review')."""
    names = gates["gate_inputs"]
    thr = gates["thresholds"][("learned", True, cfg.eps_primary)].value
    tree = gates["learned"].tree
    enc = encode_signals(names)
    specs = signal_specs(names, {n: _signal_label(td, n) for n in names})
    calib_a = td.part("calibration")

    def leaf(dist, classes):
        p = float(dist[classes.index("True")]) if "True" in classes else 0.0
        return ("auto" if p >= thr else "review"), p

    t = tree_to_table(tree, enc, specs, f"{td.task.name}_gate", "Confidence gate",
                      X_train=td.signals.loc[calib_a, names].to_numpy(), leaf_output=leaf, outputs=("auto", "review"))
    return merge_rules(t).drop_unused_inputs()


def _signal_label(td: TaskData, name: str) -> str:
    qid, kind = name.split("__")
    return f"{qid} ({ {'p': 'probability', 'conf': 'confidence', 'margin': 'margin'}[kind] })"


def outcome_table(td: TaskData, business: DecisionTable, gate: DecisionTable) -> DecisionTable:
    """Final routing: automatic only when the gate says auto and the table agrees with Jev."""
    hq = td.task.holistic.question.id
    g = InputSpec(gate.name, "Confidence gate", Domain("enum", ("auto", "review")))
    a = InputSpec(f"{business.name} = {hq}", "Table agrees with Jev", Domain("bool", (False, True)))
    rules = [Rule({g.name: frozenset({"auto"}), a.name: frozenset({True})}, "automatic"),
             Rule({g.name: frozenset({"auto"}), a.name: frozenset({False})}, "manual"),
             Rule({g.name: frozenset({"review"})}, "manual")]
    return DecisionTable(f"{td.task.name}_route", "Automate or review", [g, a], ("automatic", "manual"), rules)


# RQ4: baselines ----------------------------------------------------------------------------

def tfidf_lr(td: TaskData, ids: pd.Index, seed: int) -> np.ndarray:
    text = "narrative" if "narrative" in td.frame else "posting"
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    X = vec.fit_transform(td.frame.loc[ids, text])
    clf = LogisticRegression(C=4.0, max_iter=3000, random_state=seed).fit(X, td.y.loc[ids])
    return clf.predict(vec.transform(td.frame.loc[td.part("test"), text]))


def learning_curves(td: TaskData, schema: Schema, cfg: Config) -> pd.DataFrame:
    pool, test = td.part("pool"), td.part("test")
    rows = []
    for budget in cfg.label_budgets:
        reps = 1 if budget is None else cfg.budget_repeats
        for rep in range(reps):
            rng = np.random.default_rng(cfg.seed + rep)
            ids = pool if budget is None else pd.Index(rng.choice(pool.to_numpy(), size=min(budget, len(pool)), replace=False))
            if td.y.loc[ids].nunique() < 2:
                continue
            sup = distil(td, schema, Config(**{**asdict(cfg), "seed": cfg.seed + rep}), target="gt", ids=ids)
            out = sup["table"].evaluate(td.typed.loc[test])
            rows.append({"method": "DMN-Sup", "labels": len(ids), "rep": rep, "n_rules": sup["table"].n_rules(),
                         **metrics.accuracy(td.y.loc[test], out)})
            pred = tfidf_lr(td, ids, cfg.seed + rep)
            rows.append({"method": "TF-IDF+LR", "labels": len(ids), "rep": rep, "n_rules": np.nan,
                         **metrics.accuracy(td.y.loc[test], pred)})
    return pd.DataFrame(rows)


def operational(td: TaskData, table: DecisionTable, review_rate: float) -> dict:
    test = td.part("test")
    t0 = time.perf_counter()
    for _ in range(20):
        table.evaluate(td.typed.loc[test])
    dmn_us = (time.perf_counter() - t0) / (20 * len(test)) * 1e6
    lat = td.usage["latency_ms"]
    tokens = td.usage["input_tokens"]
    return {"jev_latency_ms_median": float(lat.median()), "jev_latency_ms_p95": float(lat.quantile(0.95)),
            "dmn_eval_us_per_case": float(dmn_us), "input_tokens_mean": float(tokens.mean()),
            "cost_usd_per_1000": float(tokens.mean() * 1000 * PRICE_PER_MTOK / 1e6),
            "review_per_1000": float(1000 * review_rate)}


# Orchestration -----------------------------------------------------------------------------

@dataclass
class TaskResult:
    task: str
    distillation: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    gate: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    operational: dict = field(default_factory=dict)
    artefacts: dict = field(default_factory=dict)


def run_task(td: TaskData, schema: Schema, cfg: Config) -> TaskResult:
    res = TaskResult(td.task.name)
    test, calib = td.part("test"), td.part("calibration")
    classes = list(td.task.classes)
    positive = td.task.holistic.true_class

    # RQ1
    d = distil(td, schema, cfg, target="jev")
    table = d["table"]
    dmn_out = pd.Series(table.evaluate(td.typed), index=td.typed.index)
    res.distillation = {
        "choice": d["choice"], "n_rules": table.n_rules(), "n_inputs": len(table.inputs),
        "mean_conditions": table.mean_conditions(),
        "fidelity_test": metrics.agreement(dmn_out.loc[test], td.jev_pred.loc[test]),
        "fidelity_ci": metrics.bootstrap_ci(lambda a, b: float(np.mean(a == b)), dmn_out.loc[test], td.jev_pred.loc[test]),
        "accuracy_test": metrics.accuracy(td.y.loc[test], dmn_out.loc[test], positive),
        "sweep": d["sweep"], "curve": d["curve"],
    }
    res.verification = table_report(d["raw_table"], table)

    # RQ2
    iso = cal.TopLabelIsotonic().fit(*_top(td, calib))
    res.calibration = {"holistic_test": cal.summary(td.jev_dist.loc[test], td.y.loc[test], iso),
                       "holistic_calib": cal.summary(td.jev_dist.loc[calib], td.y.loc[calib]),
                       "reliability_raw": cal.reliability(*_top(td, test)),
                       "reliability_iso": cal.reliability(iso.transform(_top(td, test)[0]), _top(td, test)[1])}

    # RQ3
    g = run_gates(td, dmn_out, table, cfg)
    s_learned_agree = g["scores"]["learned"].where(g["agree"], -np.inf)
    s_conf_agree = g["scores"]["jev_confidence"].where(g["agree"], -np.inf)
    ids_eval = calib.append(test)
    res.gate = {"table": g["table"], "aurc": g["aurc"], "curves": g["curves"],
                "validity": [validity_check(s, g["correct"], ids_eval, e, cfg)
                             for s in (s_conf_agree,) for e in cfg.eps_grid],
                "agreement_rate_test": float(g["agree"].loc[test].mean()),
                "accuracy_when_agree": float(g["correct"].loc[test][g["agree"].loc[test]].mean()),
                "accuracy_when_disagree": float(g["correct"].loc[test][~g["agree"].loc[test]].mean())
                if (~g["agree"].loc[test]).any() else float("nan")}
    gtab = gate_table(td, g, cfg)
    otab = outcome_table(td, table, gtab)

    # RQ4
    jev_test = td.jev_pred.loc[test]
    res.baselines = {
        "jev_holistic": metrics.accuracy(td.y.loc[test], jev_test, positive),
        "dmn_distil": metrics.accuracy(td.y.loc[test], dmn_out.loc[test], positive),
        "mcnemar_dmn_vs_jev": metrics.mcnemar(td.y.loc[test], dmn_out.loc[test], jev_test),
        "majority_class": metrics.accuracy(td.y.loc[test], np.full(len(test), td.y.loc[td.part('pool')].mode()[0]), positive),
        "learning_curves": learning_curves(td, schema, cfg),
        "classes": classes,
    }
    primary = g["table"].query("gate == 'learned' and agreement and eps == @cfg.eps_primary").iloc[0]
    res.operational = operational(td, table, 1 - float(primary["coverage"]))
    res.artefacts = {"business": table, "gate": gtab, "outcome": otab}
    return res


def _top(td: TaskData, ids: pd.Index):
    conf, _, correct = cal.top_label(td.jev_dist.loc[ids], td.y.loc[ids])
    return conf, correct
