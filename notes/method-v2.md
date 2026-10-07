# Method revision v2 (2026-10-07) — frozen

Method v1 was run once on all tasks (commit 75f146c for CFPB, 3f082de for EMSCAD). Adding EMSCAD (4.2 % fraud)
exposed two generic defects for imbalanced tasks. v2 fixes them with rules that apply to every task; nothing is
set per task. After v2 the method is frozen: no further changes are made in response to test results.

## Defects found in v1

1. **Trivial tables.** Tree selection kept the smallest tree whose validation agreement was within 0.01 of the
   best. With a rare class, the one-rule majority table is always within 0.01, so EMSCAD gave 2 rules on 1 input
   (κ = 0.21 with Jev) and the label-trained table had 0 inputs.
2. **Vacuous risk bound.** The gate bounded only the overall error rate of automated cases. With 4.2 % fraud,
   automating every posting as legitimate already meets ε = 5 %, so the bound says nothing about missed frauds.

## Changes in v2

| Component | v1 | v2 | Reference |
|---|---|---|---|
| Tree fitting | unweighted | balanced class weights | standard for imbalanced classification |
| Tree selection | smallest within 0.01 of best agreement | smallest within one standard error of the best balanced agreement (never below 0.01) | one-SE rule, Breiman et al. (1984) |
| Risk control | selective risk ≤ ε (LTT, δ) | selective risk ≤ ε **and** critical-class miss rate ≤ ε, each family at δ/2 (Bonferroni), more conservative threshold | LTT (Angelopoulos et al.); class-conditional risk |
| Critical class | — | positive class of every binary task (relief: `relief`, EMSCAD: `fraudulent`); none for multi-class routing | |
| Learned gate | unweighted tree on correctness | critical-class cases get balanced sample weight | |
| Variant | label-trained table only in learning curves | DMN-Sup (same tree procedure on pool ground truth over Jev's typed facts) gated and reported as a variant | separates Jev as feature extractor from Jev as teacher |
| Metrics | accuracy, macro-F1, F1 positive | + balanced accuracy, recall and precision of the positive class, test miss rate | |

Miss rate = (critical-class cases decided automatically and wrongly) / (critical-class cases).
Certifying any threshold needs at least ⌈log(δ/2) / log(1 − ε)⌉ critical cases in calibration split B
(ε = 0.01: 368, 0.02: 183, 0.05: 72, 0.10: 36 at δ = 0.05).

## Consequences seen after v2 (reported, not acted on)

- EMSCAD: both tables are non-trivial (Jev-distilled 49 rules / 11 inputs; DMN-Sup 23 rules / 8 inputs). Split B
  holds 62 frauds, below the 72 needed for ε = 5 %; at ε = 10 % DMN-Sup is certified (coverage 12.7 %, risk 0.8 %,
  miss rate 0.8 % on test), the Jev-distilled table is not.
- Relief: the cost-sensitive gate certifies no threshold at ε ≤ 10 % (v1 automated 11.1 % at ε = 10 %).
- Routing (no critical class): learned gate at ε = 10 % automates 40.5 % (risk 7.8 %); DMN-Sup 51.7 % (9.2 %).
