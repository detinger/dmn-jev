# EMSCAD schema pilot, v1 (2026-10-07)

Schema `configs/emscad.yaml` = `configs/emscad.v1.yaml`, sha256 `e9eb9de7bdde4b502ac244db681b4b009f27635b206e98d0e2b5bd4d5e60c2a1`.
Data: 14,562 postings after removing short texts and duplicate descriptions (611 fraudulent); splits 60/20/20
stratified by label (calibration 122 fraudulent, test 123).

Pilot (pool only): 500 random pool cases (`responses_emscad_jev_pilot.jsonl`, only 9 fraudulent) plus 150 random
fraudulent pool cases (`--pilot-class fraudulent`, `responses_emscad_jev_pilot_fraudulent.jsonl`); 646 distinct cases,
155 fraudulent. No API errors; median latency 310 ms; median 1,500 input tokens.

| Question | P(yes) legit | P(yes) fraud | AUC (p vs label) |
|---|---|---|---|
| company_described | 0.921 | 0.529 | 0.227 |
| company_named | 0.835 | 0.535 | 0.293 |
| remote_work | 0.047 | 0.097 | 0.498 |
| pay_stated | 0.246 | 0.484 | 0.590 |
| easy_money | 0.006 | 0.116 | 0.611 |
| no_experience | 0.000 | 0.077 | 0.604 |
| requirements_listed | 0.953 | 0.768 | 0.423 |
| benefits_listed | 0.273 | 0.310 | 0.538 |
| payment_required | 0.000 | 0.000 | 0.578 |
| personal_data_request | 0.004 | 0.000 | 0.695 |
| offplatform_contact | 0.004 | 0.013 | 0.556 |
| fraud_holistic | 0.102 | 0.284 | 0.684 |

job_category: admin 6 % vs 19 %, energy_industry 5 % vs 19 %, it_engineering 37 % vs 13 % (legit vs fraud).
Facts → label, 5-fold logistic regression on the pilot only: AUC 0.838. The holistic answer is weaker (AUC 0.684;
recall 0.28 at a false-positive rate of 0.10).

Decision: freeze v1 unchanged. payment_required, personal_data_request and offplatform_contact are almost never
true: EMSCAD frauds are mostly fake postings rather than advance-fee scams. The wording is literal and correct;
rewording towards the labels would fit the schema to the data, and constant inputs are simply not used by the tree.
