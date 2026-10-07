"""Join the blind routing audit annotations with the key and summarise label noise.

    python scripts/label_audit_analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

D = Path("notes/label-audit")


def fits(row, col: str) -> bool:
    """True if the queue in `col` is the annotated primary queue or listed as acceptable."""
    acc = set(str(row.acceptable).split(";")) if isinstance(row.acceptable, str) else set()
    return row[col] == row.primary or row[col] in acc


def main() -> None:
    a = pd.read_csv(D / "annotations_llm.csv").merge(pd.read_csv(D / "key.csv"), on="audit_no")
    a["group"] = (a.jev == a.queue).map({True: "agree", False: "disagree"})
    a["ann_eq_cfpb"] = a.primary == a.queue
    a["ann_eq_jev"] = a.primary == a.jev
    a["cfpb_ok"] = a.apply(fits, axis=1, col="queue")
    a["jev_ok"] = a.apply(fits, axis=1, col="jev")
    out = {}
    for g, d in a.groupby("group"):
        out[g] = {
            "n": len(d),
            "primary=CFPB": round(d.ann_eq_cfpb.mean(), 3),
            "primary=Jev": round(d.ann_eq_jev.mean(), 3),
            "primary=neither": round((~d.ann_eq_cfpb & ~d.ann_eq_jev).mean(), 3),
            "CFPB acceptable": round(d.cfpb_ok.mean(), 3),
            "Jev acceptable": round(d.jev_ok.mean(), 3),
            "clear cases": int((d.clarity == "clear").sum()),
            "clear: primary=Jev": round(d[d.clarity == "clear"].ann_eq_jev.mean(), 3),
            "clear: primary=CFPB": round(d[d.clarity == "clear"].ann_eq_cfpb.mean(), 3),
        }
    dis = a[a.group == "disagree"]
    out["disagree_by_pair"] = (dis.queue + " -> " + dis.jev).value_counts().head(8).to_dict()
    out["disagree_pair_primary_eq_jev"] = dis.groupby(dis.queue + " -> " + dis.jev).ann_eq_jev.agg(["size", "mean"]) \
        .sort_values("size", ascending=False).head(8).round(3).to_dict("index")
    # Projection to the confident population (test, conf >= .99): 892 agree, 165 disagree.
    n_agree, n_dis = 1057 - 165, 165
    p_dis_label_wrong = 1 - out["disagree"]["CFPB acceptable"]
    p_agr_label_wrong = 1 - out["agree"]["CFPB acceptable"]
    out["projection"] = {
        "observed_error_vs_cfpb": round(n_dis / 1057, 3),
        "error_vs_audit_if_jev_judged_by_acceptable": round(
            (n_dis * (1 - out["disagree"]["Jev acceptable"]) + n_agree * (1 - out["agree"]["Jev acceptable"])) / 1057, 3),
        "share_of_population_with_unacceptable_cfpb_label": round(
            (n_dis * p_dis_label_wrong + n_agree * p_agr_label_wrong) / 1057, 3),
    }
    (D / "results.json").write_text(json.dumps(out, indent=2))
    a.to_csv(D / "joined.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
