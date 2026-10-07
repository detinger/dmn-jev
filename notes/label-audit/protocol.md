# Routing label audit: protocol

**Purpose.** Estimate how many of the routing "errors" on which Jev is confident are errors in the CFPB-derived
queue label rather than in Jev. The audit measures label quality only; nothing in the method, schema or gate is
changed as a result.

**Population.** Routing test cases (n = 2,000) whose Jev queue confidence is ≥ 0.99 (n = 1,057; 165 of them disagree
with the CFPB queue).

**Sample.** `scripts/label_audit_sample.py`, random_state 2026: 150 disagreements and 50 agreements, shuffled into
audit numbers 1–200. `blind_sheet.csv` holds only the audit number and the text Jev saw. `key.csv` (CFPB queue, Jev
answer, issue, sub-issue) is not opened until annotation is finished.

**Annotator.** LLM-assisted: Claude (Opus 5.5) annotates all 200 cases from `blind_sheet.csv` only. A random subset is
re-annotated by an author, blind to both the LLM annotation and the key (`human_check.csv`). Agreement is reported as
Cohen's κ. The paper must describe the audit as LLM-assisted with human verification.

**Question.** "Using the queue definitions below, which team should handle this complaint?" The definitions are the
ones given to Jev in `configs/cfpb.yaml` (task `routing`):

| Queue | Definition |
|---|---|
| credit_reporting | Disputes about credit reports and consumer reporting agencies |
| debt_collection | Problems with debt collectors and collection of debts |
| deposits | Problems with checking or savings accounts |
| cards | Problems with credit cards or prepaid cards |
| payments | Problems with money transfers, payment apps, or virtual currency |
| mortgage | Problems with mortgages |
| consumer_loans | Problems with vehicle, student, payday, or personal loans |
| fraud | Fraud, identity theft, or transactions and accounts the consumer did not authorize |

**Rules.**
1. Choose the queue for the consumer's main problem, i.e. what they want fixed.
2. Fraud takes precedence when the main problem is a scam, identity theft, or a transaction/account the consumer
   says they did not make or authorize, whatever the product.
3. A dispute about a purchase the consumer made (merchant did not deliver, refund not received) is not fraud.
4. `primary`: one queue. `acceptable`: other queues a competent handler could reasonably choose (may be empty).
   `clarity`: `clear` if only the primary queue fits, `ambiguous` otherwise.

**Outcome measures** (after joining with `key.csv`): share of disagreements where the annotation agrees with Jev,
with CFPB, with neither; same for agreements as a control; estimated label-noise rate in the confident population.
