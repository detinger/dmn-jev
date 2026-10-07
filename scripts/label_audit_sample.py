"""Draw the blind routing label audit sample (test cases where Jev's queue confidence >= 0.99).

150 cases where Jev disagrees with the CFPB queue and 50 where it agrees, shuffled. The blind sheet holds
only an audit number and the text Jev saw; the key (labels, Jev answer) is written to a separate file.

    python scripts/label_audit_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevdmn.data.cfpb import state_for  # noqa: E402
from jevdmn.schema import load_schema  # noqa: E402

OUT = Path("notes/label-audit")


def main() -> None:
    schema = load_schema("configs/cfpb.yaml")
    answers = {}
    with open("data/interim/responses_jev.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            answers[r["case_id"]] = r["response"]["answers"]["queue_holistic"]
    s = pd.read_parquet("data/interim/cfpb_sample.parquet")
    s["case_id"] = s.complaint_id.astype(str)
    t = s[s.split_routing == "test"].copy()
    t["jev"] = [answers[c]["choice"] for c in t.case_id]
    t["jev_conf"] = [answers[c]["confidence"] for c in t.case_id]
    t = t[t.jev_conf >= 0.99]
    dis = t[t.jev != t.queue].sample(n=150, random_state=2026)
    agr = t[t.jev == t.queue].sample(n=50, random_state=2026)
    a = pd.concat([dis, agr]).sample(frac=1, random_state=2026).reset_index(drop=True)
    a["audit_no"] = range(1, len(a) + 1)
    a["text"] = [state_for(r._asdict(), schema.max_words)["complaint"] for r in a.itertuples()]
    OUT.mkdir(parents=True, exist_ok=True)
    a[["audit_no", "text"]].to_csv(OUT / "blind_sheet.csv", index=False)
    a[["audit_no", "case_id", "queue", "jev", "jev_conf", "issue", "sub_issue"]].to_csv(OUT / "key.csv", index=False)
    print(f"population (test, conf>=0.99): {len(t)}, of which disagree {int((t.jev != t.queue).sum())}; sample {len(a)}")


if __name__ == "__main__":
    main()
