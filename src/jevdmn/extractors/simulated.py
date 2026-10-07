"""Simulated System One extractor for developing and testing the pipeline offline.

It produces responses in the Jev format from keyword cues in the narrative and
from the ground-truth labels, with noise. Its outputs are NOT model results and
must never be reported as such; the reporting layer refuses simulated runs.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from .base import Extraction

CUES = {
    "unauthorized": r"unauthori[sz]ed|did not authori[sz]e|never authori[sz]ed|without my (permission|consent|knowledge)",
    "identity_theft": r"identity theft|stole my identity|victim of identity|fraudulent(ly)? opened|not my account",
    "scam": r"\bscam|scammer|tricked|fraudster|impersonat",
    "report_dispute": r"(credit report|reporting|equifax|experian|transunion).{0,200}(dispute|inaccurate|incorrect|remove|delete)",
    "collector_contact": r"collector|collection agency|collections? (calls|letters)|calling me",
    "money_lost": r"\{\$[\d,\.]+\}|charged|lost|stolen|withdrawn|refund",
    "fee_or_interest": r"\bfees?\b|interest|penalt|overdraft",
    "refund_requested": r"refund|reimburs|money back|return (of )?my|credit (it )?back",
    "prior_contact": r"called|contacted|spoke (to|with)|emailed|reached out|customer service",
    "company_admitted": r"admitted|apologi[sz]ed|acknowledged (the|their) (error|mistake)|promised",
    "claim_denied": r"denied|rejected|declined|closed (my|the) (dispute|claim)",
    "funds_blocked": r"froze|frozen|on hold|restricted|closed my account|locked",
}

PRODUCT_OPTION = {
    "Credit reporting or other personal consumer reports": "credit_report",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_report",
    "Debt collection": "debt_collection",
    "Checking or savings account": "bank_account",
    "Credit card": "credit_card",
    "Credit card or prepaid card": "credit_card",
    "Prepaid card": "prepaid_card",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Mortgage": "mortgage",
    "Vehicle loan or lease": "vehicle_loan",
    "Student loan": "student_loan",
    "Payday loan, title loan, personal loan, or advance loan": "personal_loan",
    "Payday loan, title loan, or personal loan": "personal_loan",
}


def _rng(case_id: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(case_id.encode()).hexdigest()[:12], 16))


def _peaked(options: list[str], true: str, rng, accuracy: float, sharpness: float) -> dict:
    k = len(options)
    pick = true if rng.random() < accuracy else options[rng.integers(k)]
    alpha = np.full(k, 0.3)
    alpha[options.index(pick)] = sharpness
    p = rng.dirichlet(alpha)
    return {o: float(round(v, 4)) for o, v in zip(options, p)}


def _choice_answer(probs: dict) -> dict:
    n = len(probs)
    top = max(probs, key=probs.get)
    conf = (probs[top] - 1 / n) / (1 - 1 / n)
    return {"type": "choice", "choice": top, "probabilities": probs, "confidence": round(conf, 4)}


def _noul(cue: bool, rng, base: float = 0.1) -> float:
    logit = (2.2 if cue else -2.2) + rng.normal(0, 1.4) + np.log(base / (1 - base)) * 0.2
    return float(round(1 / (1 + np.exp(-logit)), 4))


class SimulatedExtractor:
    name = "simulated"

    def __init__(self, records: dict[str, dict]):
        self.records = records  # case_id -> corpus row (with ground truth)

    def extract(self, cases: Iterable[tuple[str, dict]], questions: dict) -> list[Extraction]:
        return [self._one(cid, state, questions) for cid, state in cases]

    def _one(self, case_id: str, state: dict, questions: dict) -> Extraction:
        rng = _rng(case_id)
        rec = self.records[case_id]
        text = str(state.get("complaint", "")).lower()
        cues = {k: bool(re.search(v, text)) for k, v in CUES.items()}
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice" and qid == "product":
                true = PRODUCT_OPTION.get(rec["product"], "other")
                answers[qid] = _choice_answer(_peaked(list(q["criteria"]), true, rng, 0.88, 8.0))
            elif q["type"] == "choice":  # holistic queue
                answers[qid] = _choice_answer(_peaked(list(q["criteria"]), rec["queue"], rng, 0.78, 5.0))
            elif q["type"] == "noul" and qid in CUES:
                answers[qid] = {"type": "noul", "noul": _noul(cues[qid], rng)}
            elif q["type"] == "noul":  # holistic relief
                signal = 1.3 * cues["unauthorized"] + 1.0 * cues["refund_requested"] + 0.8 * cues["money_lost"] \
                    + 0.9 * (rec["relief"] == "relief") - 2.0
                p = 1 / (1 + np.exp(-(signal + rng.normal(0, 1.0))))
                answers[qid] = {"type": "noul", "noul": float(round(p, 4))}
            elif q["type"] == "score":
                k = len(q["criteria"])
                level = min(k - 1, int(cues["money_lost"]) + int(cues["funds_blocked"]) + int(rng.random() < 0.2))
                probs = _peaked([str(i) for i in range(k)], str(level), rng, 0.85, 6.0)
                p = np.array([probs[str(i)] for i in range(k)])
                m = int(p.argmax())
                spread = float((p * np.abs(np.arange(k) - m)).sum())
                even = float(np.abs(np.arange(k) - (k - 1) / 2).mean())
                answers[qid] = {"type": "score", "score": float(round((p * np.arange(k)).sum(), 4)),
                                "probabilities": probs, "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                "confidence": round(max(0.0, 1 - spread / even), 4)}
        latency = float(rng.lognormal(np.log(150), 0.35))
        response = {"model": "simulated-0", "answers": answers, "usage": {"input_tokens": len(text.split()) * 4 // 3 + 900, "output_tokens": 0}}
        return Extraction(case_id=case_id, response=response, latency_ms=latency,
                          timestamp=datetime.now(timezone.utc).isoformat(), extractor="simulated")
