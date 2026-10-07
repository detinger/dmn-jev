"""CFPB consumer complaints: corpus construction, labels, sampling and splits.

Narratives come from a public mirror of the CFPB Consumer Complaint Database
(complaints up to 2025-01-02), because the CFPB stopped publishing narratives
on 2026-09-14. Final company responses come from the official bulk snapshot,
joined on the complaint id, so complaints that were still "In progress" in the
mirror receive their final outcome.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RAW = Path("data/raw")
MIRROR_DIR = RAW / "hf_cfpb"
OFFICIAL_ZIP = RAW / "complaints.csv.zip"
DATE_MIN, DATE_MAX = "2023-04-01", "2025-01-01"

PRODUCT_GROUP = {
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Debt collection": "debt_collection",
    "Checking or savings account": "bank_account",
    "Credit card": "card",
    "Prepaid card": "card",
    "Credit card or prepaid card": "card",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Mortgage": "mortgage",
    "Vehicle loan or lease": "consumer_loans",
    "Student loan": "consumer_loans",
    "Payday loan, title loan, personal loan, or advance loan": "consumer_loans",
    "Payday loan, title loan, or personal loan": "consumer_loans",
}

# Handling queue for every product group when the complaint is not fraud-related.
QUEUE_BY_GROUP = {
    "credit_reporting": "credit_reporting",
    "debt_collection": "debt_collection",
    "bank_account": "deposits",
    "card": "cards",
    "money_transfer": "payments",
    "mortgage": "mortgage",
    "consumer_loans": "consumer_loans",
}

# (issue, sub_issue) pairs routed to the fraud queue; None matches any sub-issue.
FRAUD_ISSUES = {
    ("Attempts to collect debt not owed", "Debt was result of identity theft"),
    ("Fraud or scam", None),
    ("Problem with a lender or other company charging your account", "Transaction was not authorized"),
    ("Problem with fraud alerts or security freezes", None),
    ("Getting a credit card", "Card opened without my consent or knowledge"),
    ("Getting a credit card", "Card opened as result of identity theft or fraud"),
    ("Unauthorized transactions or other transaction problem", None),
    ("Opening an account", "Account opened without my consent or knowledge"),
    ("Opening an account", "Account opened as a result of fraud"),
    ("Getting a loan or lease", "Loan opened without my consent or knowledge"),
    ("Getting a loan or lease", "Fraudulent loan"),
    ("Getting a loan", "Loan opened without my consent or knowledge"),
    ("Getting a loan", "Fraudulent loan"),
    ("Unauthorized withdrawals or charges", None),
}

RELIEF_GROUPS = ("bank_account", "card", "money_transfer")
CLOSED = ("Closed with explanation", "Closed with non-monetary relief", "Closed with monetary relief")


def is_fraud(issue: str | None, sub_issue: str | None) -> bool:
    return (issue, sub_issue) in FRAUD_ISSUES or (issue, None) in FRAUD_ISSUES


def normalise(text: str) -> str:
    text = re.sub(r"X{2,}", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_mirror() -> pd.DataFrame:
    cols = ["date_received", "product", "sub_product", "issue", "sub_issue",
            "complaint_what_happened", "company", "company_response", "complaint_id"]
    parts = []
    for f in sorted(MIRROR_DIR.glob("*.parquet")):
        t = pq.read_table(f, columns=cols)
        keep = pc.and_(pc.greater_equal(t["date_received"], DATE_MIN), pc.less(t["date_received"], DATE_MAX))
        t = t.filter(keep)
        narrative = pc.fill_null(t["complaint_what_happened"], "")
        parts.append(t.filter(pc.greater(pc.utf8_length(narrative), 0)))
    df = pa.concat_tables(parts).to_pandas()
    df = df.rename(columns={"complaint_what_happened": "narrative", "company_response": "response_mirror"})
    df["complaint_id"] = df["complaint_id"].astype("int64")
    return df


def load_official_responses() -> pd.DataFrame:
    usecols = ["Complaint ID", "Product", "Issue", "Sub-issue", "Company response to consumer"]
    with zipfile.ZipFile(OFFICIAL_ZIP) as z, z.open("complaints.csv") as fh:
        chunks = pd.read_csv(fh, usecols=usecols, dtype=str, chunksize=1_000_000)
        out = pd.concat(chunks, ignore_index=True)
    out = out.rename(columns={"Complaint ID": "complaint_id", "Product": "product_official",
                              "Issue": "issue_official", "Sub-issue": "sub_issue_official",
                              "Company response to consumer": "response"})
    out["complaint_id"] = out["complaint_id"].astype("int64")
    return out


def build_corpus(min_words: int = 30) -> pd.DataFrame:
    """All eligible complaints with labels for both decision tasks."""
    df = load_mirror()
    official = load_official_responses()
    df = df.merge(official, on="complaint_id", how="inner")
    df = df[df["response"].isin(CLOSED)]
    df["product_group"] = df["product"].map(PRODUCT_GROUP)
    df = df[df["product_group"].notna()]

    df["n_words"] = df["narrative"].str.split().str.len()
    df = df[df["n_words"] >= min_words]
    # Templated letters (common in credit-repair complaints) would leak across splits.
    df["text_hash"] = df["narrative"].map(lambda s: hashlib.sha1(normalise(s).encode()).hexdigest())
    df = df.drop_duplicates("text_hash", keep="first")

    df["fraud"] = [is_fraud(i, s) for i, s in zip(df["issue"], df["sub_issue"].where(df["sub_issue"].notna(), None))]
    df["queue"] = np.where(df["fraud"], "fraud", df["product_group"].map(QUEUE_BY_GROUP))
    df["relief"] = np.where(df["response"] == "Closed with monetary relief", "relief", "no_relief")
    df["label_changed"] = df["response"] != df["response_mirror"]
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class SampleSpec:
    relief_n: int = 8000
    routing_per_queue: int = 1250
    seed: int = 20261007
    split: tuple[float, float, float] = (0.6, 0.2, 0.2)  # pool / calibration / test


def draw_sample(corpus: pd.DataFrame, spec: SampleSpec = SampleSpec()) -> pd.DataFrame:
    """Union of a random relief sample (banking products) and a queue-stratified routing sample.

    Each case carries a split assignment per task. Splits are stratified by the
    task's ground truth; the pool split is used without its labels (distillation).
    """
    rng = np.random.default_rng(spec.seed)
    banking = corpus[corpus["product_group"].isin(RELIEF_GROUPS)]
    relief_ids = rng.choice(banking["complaint_id"].to_numpy(), size=spec.relief_n, replace=False)

    routing_ids = []
    for _, grp in corpus.groupby("queue", sort=True):
        n = min(spec.routing_per_queue, len(grp))
        routing_ids.extend(rng.choice(grp["complaint_id"].to_numpy(), size=n, replace=False))

    ids = np.union1d(relief_ids, np.asarray(routing_ids))
    sample = corpus[corpus["complaint_id"].isin(ids)].copy()
    sample["in_relief"] = sample["complaint_id"].isin(relief_ids)
    sample["in_routing"] = sample["complaint_id"].isin(routing_ids)
    sample["split_relief"] = _stratified_split(sample, "in_relief", "relief", spec, rng)
    sample["split_routing"] = _stratified_split(sample, "in_routing", "queue", spec, rng)
    return sample.sort_values("complaint_id").reset_index(drop=True)


def _stratified_split(df: pd.DataFrame, member_col: str, label_col: str, spec: SampleSpec, rng) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    names = ("pool", "calibration", "test")
    for _, grp in df[df[member_col]].groupby(label_col):
        idx = rng.permutation(grp.index.to_numpy())
        cuts = np.floor(np.cumsum(spec.split)[:-1] * len(idx)).astype(int)
        for name, chunk in zip(names, np.split(idx, cuts)):
            out.loc[chunk] = name
    return out


def state_for(record: pd.Series | dict, max_words: int) -> dict:
    words = str(record["narrative"]).split()
    text = " ".join(words[:max_words])
    return {"complaint": text}
