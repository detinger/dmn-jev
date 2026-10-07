"""EMSCAD job postings (Vidros et al., 2017): corpus, labels, sampling and splits.

Source: the original EMSCAD site (emscad.samos.aegean.gr) is offline; the CSV comes from the Hugging Face
mirror victor/real-or-fake-fake-jobposting-prediction (17,880 postings, 866 fraudulent, as published).
Its sha256 is stored next to the file.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/emscad/fake_job_postings.csv")
CASE_ID = "case_id"
SEED = 20261007
SPLIT = (0.6, 0.2, 0.2)  # pool / calibration / test, as for CFPB

# Text fields shown to the model, in posting order. Structured metadata (logo, screening questions,
# telecommuting flag) is withheld, as the CFPB structured fields are.
FIELDS = (("title", "Title"), ("location", "Location"), ("salary_range", "Salary"),
          ("employment_type", "Employment type"), ("company_profile", "Company profile"),
          ("description", "Description"), ("requirements", "Requirements"), ("benefits", "Benefits"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def build_corpus(min_words: int = 30) -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["posting"] = df.apply(_posting_text, axis=1)
    df["n_words"] = df["posting"].str.split().str.len()
    df = df[df["n_words"] >= min_words]
    # The same posting is often listed for several locations; near-identical texts would leak across splits.
    df["text_hash"] = df["description"].fillna("").map(lambda s: hashlib.sha1(_norm(s).encode()).hexdigest())
    df = df.drop_duplicates("text_hash", keep="first")
    df[CASE_ID] = "job" + df["job_id"].astype(str)
    df["label"] = np.where(df["fraudulent"] == 1, "fraudulent", "legitimate")
    return df.reset_index(drop=True)


def draw_sample(corpus: pd.DataFrame) -> pd.DataFrame:
    """All postings; pool / calibration / test stratified by the label."""
    rng = np.random.default_rng(SEED)
    s = corpus.copy()
    s["in_jobfraud"] = True
    s["split_jobfraud"] = pd.Series(pd.NA, index=s.index, dtype="object")
    for _, grp in s.groupby("label"):
        idx = rng.permutation(grp.index.to_numpy())
        cuts = np.floor(np.cumsum(SPLIT)[:-1] * len(idx)).astype(int)
        for name, chunk in zip(("pool", "calibration", "test"), np.split(idx, cuts)):
            s.loc[chunk, "split_jobfraud"] = name
    return s.sort_values(CASE_ID).reset_index(drop=True)


def _posting_text(r) -> str:
    parts = [f"{label}: {r[col]}" for col, label in FIELDS if isinstance(r[col], str) and r[col].strip()]
    return "\n".join(parts)


def state_for(record: pd.Series | dict, max_words: int) -> dict:
    lines, n, out = str(record["posting"]).split("\n"), 0, []
    for line in lines:
        words = line.split()
        if n + len(words) > max_words:
            out.append(" ".join(words[: max_words - n]))
            break
        out.append(line)
        n += len(words)
    return {"posting": "\n".join(out)}
