# Schema pilot, v1 (2026-10-07)

Schema: `configs/cfpb.yaml` = `configs/cfpb.v1.yaml`, sha256 `b08a014ba98107bdda7e4e74805212c95fd9bbd7bb6cad51ad8add6d78b3c714`.
Model: `jev-1.13.0` (all responses). Responses: `data/interim/responses_jev_pilot.jsonl`.
Analysis script: `scripts/pilot_analysis.py`.

## Smoke test (20 requests)

- `--limit 20` takes the first 20 sample rows: 12 pool, 3 calibration, 5 test. Only the 12 pool cases were read
  by hand. Calibration/test responses were not inspected and played no part in schema decisions.
  (Future smoke tests should draw from the pool.)
- Response format matches `features.py`: score probabilities keyed `"0"`…`"3"`, `confidence` on choice/score,
  probabilities sum to 1. Score answers also carry an expected value (`"score": 1.55`), which the pipeline ignores
  (typed value = argmax).
- No 4xx/5xx errors, no retries.

## Pilot (500 pool cases, random_state=11)

Cost/latency: median 295 ms (p95 407 ms) per request at concurrency 16; input tokens median 1,350 (max 3,241),
output median 444. 500 requests took ~10 s.

### Answer distributions

| Question | share true | extreme (<.05 or >.95) | uncertain (.3–.7) |
|---|---|---|---|
| unauthorized | 0.370 | 0.190 | 0.164 |
| identity_theft | 0.138 | 0.590 | 0.066 |
| scam | 0.062 | 0.596 | 0.066 |
| report_dispute | 0.238 | 0.426 | 0.100 |
| collector_contact | 0.124 | 0.588 | 0.070 |
| money_lost | 0.536 | 0.280 | 0.176 |
| fee_or_interest | 0.140 | 0.400 | 0.060 |
| refund_requested | 0.444 | 0.278 | 0.208 |
| prior_contact | 0.748 | 0.278 | 0.124 |
| company_admitted | 0.118 | 0.382 | 0.128 |
| claim_denied | 0.554 | 0.164 | 0.232 |
| funds_blocked | 0.266 | 0.134 | 0.112 |

- No degenerate noul (none always near 0 or 1). Every choice option and every score level was used at least once.
- No first-option preference: `credit_report` (listed first) chosen 10.8 %, `credit_reporting` 13.4 %, in line
  with their base rates.
- `harm` is compressed: level 2 in 58 % of cases, level 3 in 5 %.
- High correlations: unauthorized ~ identity_theft (r = 0.68), money_lost ~ refund_requested (r = 0.67).

### Product (fact) vs CFPB product group

Good agreement. Main off-diagonals: money_transfer → bank_account (17/70; bank-app transfers, ATM deposits),
debt_collection → credit_report (12/47). Store credit cards are sometimes answered `other`.

### Routing (holistic, n = 290): agreement with CFPB queue 0.690

Main confusions (CFPB → Jev): deposits → fraud 20/48, payments → fraud 7/40, debt_collection → credit_reporting
10/32, fraud → credit_reporting 8/33. Jev predicts fraud 58 times vs 33 true.

Reading deposits → fraud cases: 6 of the 6 read describe a scam, unauthorized ATM/debit withdrawals, identity theft
or check fraud, but the consumer filed them under a non-fraud issue (e.g. "Managing an account / Problem using a
debit or ATM card"). These are mostly **label noise** from the consumer's self-chosen category, not Jev errors.
Fraud → credit_reporting is mostly "Attempts to collect debt not owed / Debt was result of identity theft" (6/8),
a case the FRAUD_ISSUES mapping sends to fraud while the narrative is about a credit report entry.

### Relief (holistic, n = 217): base rate 0.157, Jev says relief 0.618, AUC 0.727, accuracy@0.5 0.512

Jev answers the normative question ("should refund") and is far more generous than companies are. Ranking is
informative (AUC 0.73), so the calibrated gate can still work, but coverage at small ε will be low (expected in the
plan's "unfavourable results" table). Facts that separate labels: money_lost (0.52 vs 0.88), refund_requested
(0.49 vs 0.74), fee_or_interest (0.12 vs 0.29), funds_blocked (0.36 vs 0.18).

## Candidate changes for v2 (one round allowed; not applied)

1. `unauthorized` over-triggers (37 % true; only 28 % of those carry a CFPB fraud label). Top non-fraud hits are
   card purchase disputes ("isn't resolving a dispute about a purchase", 14) and account management. Wording
   should exclude disputes about purchases the consumer made, and "or account was made" is ungrammatical.
2. `product.credit_card`: name store/retail credit cards explicitly.
3. `fraud` queue criterion: **leave unchanged**. Rewording it to reproduce consumers' form choices would fit the
   wording to label noise; report the noise instead (optionally audit routing disagreements under O2).
4. `relief_holistic`: decision for the authors. Keep the normative wording (Jev as the decision maker being
   distilled; label mismatch is a finding) or ask for the likely company outcome (closer to the labels, but Jev
   then predicts company behaviour rather than deciding).
