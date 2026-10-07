"""Agreement of the human check with the LLM annotation and with CFPB/Jev.

    python scripts/label_audit_human.py [notes/label-audit/human_check.csv]
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "notes/label-audit/human_check.csv"
    h = pd.read_csv(path)[["audit_no", "primary"]].rename(columns={"primary": "human"}).dropna()
    j = pd.read_csv("notes/label-audit/joined.csv")
    m = h.merge(j, on="audit_no")
    print(f"n = {len(m)}; human = LLM: {(m.human == m.primary).mean():.3f}; "
          f"kappa = {cohen_kappa_score(m.human, m.primary):.3f}")
    d = m[m.group == "disagree"]
    print(f"disagreements n = {len(d)}: human = Jev {(d.human == d.jev).mean():.3f}, "
          f"human = CFPB {(d.human == d.queue).mean():.3f}")


if __name__ == "__main__":
    main()
