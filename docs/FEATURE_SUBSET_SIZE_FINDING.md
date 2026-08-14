# Feature subset size: an open finding against the shipped configuration

Generated 2026-08-14 from `artifacts/metrics/nested_feature_family_arms.json`
(runtime 540 s; 3 repeats x 5 outer folds = 15 paired measurements per arm).

This document exists because the experiment disagreed with the shipped model and
the honest thing to do is write that down rather than quietly not run the arm.

## What was asked

Section 21.1 asks for the feature-subset curve **either side** of the shipped
size, not only below it. Earlier runs swept 120 / 60 / 30 - every arm below the
shipped size - which can only ever confirm that cutting further hurts. It cannot
answer the question that actually matters: *did we cut too deep already?*

The arm list is now `120 (shipped), 200, 100, 60, 30`.

## What came back

Mean average precision across the 15 outer folds, and the paired difference
against the shipped `full_clean` (top-120) arm:

| arm | pool | used | AP | paired diff | sd | sign p | Wilcoxon p | paired-t p |
|---|---|---|---|---|---|---|---|---|
| `full_clean` (shipped) | 3925 | 120 | 0.78161 | - | - | - | - | - |
| **`full_clean_top200`** | 3925 | 200 | **0.80577** | **+0.02416** | 0.03169 | **0.0352** | **0.0084** | **0.0105** |
| `full_clean_top100` | 3925 | 100 | 0.77533 | -0.00628 | 0.01981 | 0.3018 | 0.3303 | 0.2401 |
| `full_clean_top60` | 3925 | 60 | 0.73103 | -0.05058 | 0.04882 | 0.0010 | 0.0002 | 0.0013 |
| `full_clean_top30` | 3925 | 30 | 0.63025 | -0.15136 | 0.11461 | 0.0001 | 0.0001 | 0.0002 |
| `no_meta_features` | 3912 | 120 | 0.78443 | +0.00283 | 0.01685 | 0.3877 | 0.9375 | 0.5265 |
| `behavior_profile` | 3905 | 120 | 0.76603 | -0.01558 | 0.02488 | 0.0352 | 0.0301 | 0.0295 |
| `behavior_only` | 3897 | 120 | 0.76260 | -0.01901 | 0.02988 | 0.1185 | 0.0301 | 0.0273 |
| `meta_features_only` | 13 | 13 | 0.16754 | -0.61407 | 0.12498 | 0.0001 | 0.0001 | 0.0000 |
| `bank_prior` | 18 | 18 | 0.28676 | -0.49485 | 0.12685 | 0.0001 | 0.0001 | 0.0000 |
| `alert_only` | 20 | 19 | 0.06336 | -0.71825 | 0.09257 | 0.0001 | 0.0001 | 0.0000 |

The pre-registered replacement rule - *positive mean paired difference and at
least 2 of 3 paired tests below p < 0.05* - is applied by the code itself, not
by a human reading the table. Its verdict for `full_clean_top200` is
`REPLACES_FULL_CLEAN`, and it clears the bar on all three tests rather than the
two it needs. Every other arm returns `NO_CHANGE`.

One incidental result worth recording: the seven arms that also ran in the
previous sweep returned **bit-identical** AP values this time (`full_clean`
0.78161, `top60` 0.73103, `top30` 0.63025, `bank_prior` 0.28676, `behavior_only`
0.76260, `behavior_profile` 0.76603, `alert_only` 0.06336). Two independent runs
on separately rebuilt folds agreeing to five decimals is the seeding and
fold-construction determinism working, and it means the two new arms can be read
against the old ones without a "different run" caveat.

## What this means, stated plainly

**The shipped feature-set size of 120 appears to be cut too deep.** Going to 200
features is worth roughly +0.024 AP, and the improvement is consistent enough
across folds to clear the significance bar that was set before the run.

Two things keep this from being a crisis rather than a finding:

- The effect is smaller than the champion's own fold-to-fold spread
  (+0.024 against a champion std of 0.027). The shipped model is not wrong; it
  is leaving something on the table.
- The curve is flat between 100 and 120 (p = 0.30). The cliff is at 60 and
  below. So this is a question of where on a plateau to sit, not evidence that
  the selection procedure is broken.

## What was *not* done about it, and why

**The champion was not re-fitted at 200 features.** Acting on this finding means
re-running selection, re-fitting, re-calibrating, re-bundling and re-validating
the champion end to end - and then every downstream artifact that quotes
0.76904, every threshold, the calibration curve, the bundle hash and the model
card all move with it. That is a retrain, and a retrain was outside the scope of
this pass.

Shipping the *number* from this arm while shipping the *model* from the old one
would be the actual dishonesty here, so neither the registry nor
`final_accuracy_table.csv` has been touched. The champion remains
`xgboost_top_120` at OOF PR-AUC 0.76904 +/- 0.02663, which is a figure that a
served, calibrated, bundled model actually produces.

## The recommendation

Re-run the champion at 200 features before the next release gate. Budget one
full nested-CV cycle. If it reproduces, promote it; the paired evidence for
doing so is already recorded here and does not need to be gathered again.

## Secondary reading: the meta-features

`no_meta_features` is statistically indistinguishable from the shipped arm
(+0.003, Wilcoxon p = 0.94). The 13 engineered meta-features are not carrying
the model - `meta_features_only` scores 0.168 on their own. They are retained
for explanation quality, not for accuracy, and this table is the evidence for
saying so out loud rather than implying they earn their place numerically.

## Reproduce

```bash
.venv/Scripts/python.exe -m muleguard.cli.nested_ses --stage families
```

Reads dev only. The run logs `firewall verified: 0 of 13 quarantined columns
present` before any arm is scored; if that line is missing the numbers above are
void.
