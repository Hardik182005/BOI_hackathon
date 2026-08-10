# Merchant Legitimacy Verifier

**Implementation:** `src/muleguard/models/merchant.py`
**Fitting and evaluation CLI:** `src/muleguard/cli/merchant_verifier.py`
(`python -m muleguard.cli.merchant_verifier`)
**Machine-readable results:** `artifacts/metrics/merchant_verifier_v2.json`
(generated 2026-08-12T08:46:23 UTC)
**Frozen model:** `artifacts/models/merchant_verifier.joblib`

---

## 1. The problem, and the shortcut this project refuses

Legitimate merchants trip mule heuristics. High inbound velocity, many
counterparties, rapid outflow, cash-out pressure — a busy retailer looks
structurally similar to a pass-through account. Something has to be done about
it, and there is a well-known shortcut:

> detect that the account is a merchant, then multiply its risk score by a
> hand-picked constant such as 0.70.

This project does not do that, for three reasons stated in the module
docstring:

1. **The constant is invented.** Nobody can say where 0.70 came from, why not
   0.60, or what changes if the book changes.
2. **It destroys the calibration.** The score leaving the model is a calibrated
   probability. Multiply it by 0.70 and it is no longer a probability of
   anything — but it is still displayed, thresholded and reported as though it
   were.
3. **It is unfalsifiable.** After a multiplier has been applied for six months,
   there is no experiment that says whether it prevented false positives or
   hid a real mule. The information needed to tell those apart was thrown away
   at the moment of multiplication.

The last one is the serious one. A mule network that routes through a business
account is not a hypothetical; business accounts are attractive to mule
operators **precisely because** they attract less scrutiny. A blind multiplier
gives that behaviour an automatic discount.

---

## 2. What replaces it

A model, fitted on business evidence alone, asked the same question the main
model is asked.

- **View:** `E_profile_merchant` — POS acceptance, GST activity, balance
  profile, loans, standing instructions, fees and charges, total-all-rails and
  tenure/profile fields (1,376 admitted candidate columns).
- **Target:** the same mule label. The verifier is not a "merchantness"
  detector. It measures **how much exculpatory weight the business evidence
  carries on its own** — how far the business picture alone fails to
  corroborate the alert.
- **Estimator:** class-weighted LightGBM (250 trees, 15 leaves, lr 0.05,
  `scale_pos_weight` set to the observed negative/positive ratio, seed 42).
- **Protocol:** fitted out-of-fold on the development split using the same
  saved folds every other model uses, so its PR-AUC is comparable with the
  tournament rather than an in-sample figure. It never sees the locked test.

The output an analyst sees is a **legitimacy percentile**, not a raw
probability: `legitimacy = 1 − percentile(model probability)` within the
development distribution. Percentiles rather than probabilities because the
bands must keep their meaning across refits — a better-calibrated model would
otherwise silently move every account's band without anyone touching a
threshold. "Strong" therefore means *stronger business evidence than 80 % of
this book*, an explicitly relative claim, not an absolute one.

| Band | Legitimacy percentile |
|---|---|
| `STRONG_BUSINESS_EVIDENCE` | ≥ 0.80 |
| `SOME_BUSINESS_EVIDENCE` | ≥ 0.50 |
| `LITTLE_BUSINESS_EVIDENCE` | ≥ 0.00 |

---

## 3. The contract: what it may and may not touch

The verifier's output influences exactly one thing —
**auto-escalation confidence** — and the separation is the point of the design.

The risk score answers *"how mule-like is this behaviour?"*. Escalation
confidence answers *"how sure are we that a machine should push this up the
queue before a person looks at it?"*. Strong business evidence lowers the
second without touching the first.

| May do | May never do |
|---|---|
| Lower the confidence attached to an automatic escalation | Modify the calibrated risk, the model score, or any threshold |
| Appear as exculpatory evidence in the ProofGraph defence panel | Remove an account from a review queue |
| | Act as a merchant whitelist |
| | Apply a fixed multiplier to any score |

`escalation_confidence()` returns the tier and the calibrated risk under the
field names `risk_tier_unchanged` and `calibrated_risk_unchanged`, and emits
three guarantees with every call:

> - no adjustment modified the calibrated risk or any threshold
> - no adjustment removed this account from a review queue
> - every adjustment above lists the value that triggered it

The strongest possible effect of a `STRONG_BUSINESS_EVIDENCE` verdict is
`HIGH → MEDIUM` (or `MEDIUM → LOW`) confidence, which routes a case to a human
**sooner**, not out of the queue. Two other factors move the same dial:
conformal abstention (`UNCERTAIN_SET`) and model disagreement below 0.5. The
disagreement rationale is explicit that uncertainty is not evidence of guilt
and does not raise the risk score.

Every adjustment is emitted with the value that triggered it, so an auditor can
reconstruct the decision from the record alone.

### The disclaimer that ships with every verdict

> Business evidence is exculpatory context, never a whitelist. A genuine
> merchant can also be a mule, and mule networks are known to use business
> accounts precisely because they attract less scrutiny. This verifier can only
> reduce the confidence attached to an automatic escalation; it cannot clear an
> account, cannot lower its risk score, and cannot remove it from a review
> queue.

---

## 4. Measured results

From `artifacts/metrics/merchant_verifier_v2.json`.

### 4.1 Out-of-fold discrimination — the defensible number

| Measure | Value |
|---|---:|
| View | `E_profile_merchant` |
| Features | 1,376 |
| Development rows / positives | 7,264 / 64 |
| Prevalence | 0.008811 |
| **OOF PR-AUC** | **0.64455** |
| OOF ROC-AUC | 0.95359 |
| Lift over prevalence | 73.2× |

This is the number that matters, because it is the one measured out of fold.
It says the business-evidence view alone carries real information about mule
risk — the verifier is not a decorative confidence knob. It is well below the
promoted behavioural model's 0.7690
(`docs/FINAL_ACCURACY_AND_MODEL_SELECTION_REPORT.md` §3), which is the expected
ordering: business context is context, not the primary signal.

**Reconciling this with the tournament.** `tournament_v2.json` records
`lightgbm_viewE_top_60` at PR-AUC 0.3061 ± 0.0842 on the same view. The two
numbers are not in conflict and neither supersedes the other — they measure
different configurations:

| | Tournament `lightgbm_viewE_top_60` | Merchant verifier |
|---|---|---|
| Features | 60 (stability-selected subset) | all 1,376 in the view |
| Repeats | 3 | 1 (`dev.fold_ids[0]`) |
| PR-AUC | 0.3061 ± 0.0842 | 0.64455 |

The verifier uses the whole view and one repeat; the tournament entry uses a
60-feature subset across three. The single-repeat figure carries no variance
estimate, and given the tournament's ± 0.08 spread on the narrower
configuration, 0.64455 should be read as one draw, not as a stable estimate.
Running the verifier over the full three repeats would tighten this, and has
not been done.

### 4.2 Band composition — in-sample, read with care

| Band | Accounts |
|---|---:|
| `STRONG_BUSINESS_EVIDENCE` | 1,452 |
| `SOME_BUSINESS_EVIDENCE` | 2,180 |
| `LITTLE_BUSINESS_EVIDENCE` | 3,632 |

| Strong-evidence band | Value |
|---|---:|
| Accounts | 1,452 |
| Labelled mules among them | **0** |
| Mule prevalence in band | 0.0 |
| Book prevalence | 0.008811 |
| Relative prevalence | 0.0 |
| Recorded verdict | `USEFUL_AS_EXCULPATORY_CONTEXT` |

**This table is in-sample and must be labelled as such.** Reading
`cli/merchant_verifier.py`: the band assignment is produced by fitting the
final verifier on the whole development split and then calling `verdicts()` on
those same rows, with the percentile reference taken from the same fit. The
model has seen every label it is being scored against. A clean "0 mules in
1,452 accounts" is exactly what an in-sample fit is expected to produce, and it
is **not** evidence that the band would be empty of mules on unseen data.

The `USEFUL_AS_EXCULPATORY_CONTEXT` verdict is derived from this in-sample
comparison (`strong_prevalence < prevalence`), so it inherits the same caveat.
The out-of-fold PR-AUC of 0.64455 supports the weaker and defensible claim that
business evidence is informative; it does not establish that the strong band is
mule-free out of sample. An out-of-fold version of the band composition is the
measurement that would settle it, and **it has not been run**.

Nothing downstream depends on the resolution: the verifier's only permitted
effect is lowering escalation confidence, so even an over-optimistic band
cannot clear an account or remove it from a queue.

### 4.3 The self-check that can disable it

The CLI computes whether mules are genuinely rarer in the strong band than in
the book, and writes one of two verdicts:

- `USEFUL_AS_EXCULPATORY_CONTEXT` — the band carries exculpatory information.
- `NOT_EXCULPATORY_DO_NOT_USE_TO_LOWER_CONFIDENCE` — with the recorded
  instruction that using it "would deprioritise exactly the accounts mule
  networks prefer. Wire the verifier as reporting-only until this changes."

A component that ships with a measured condition for its own deactivation is
the opposite of a hand-picked 0.70. The current run records the first verdict,
with the in-sample caveat in §4.2 attached.

---

## 5. Integration status

At the time of writing, `escalation_confidence()` has no caller elsewhere in
`src/` — a repository-wide search finds the definition in
`models/merchant.py` and no invocation from the scoring path, the policy engine
or the API, and no test file references it. The verifier is fitted, evaluated
and frozen; the confidence-adjustment function is implemented and its contract
is fixed, but it is **not yet wired into a live scoring response**. This is
recorded rather than implied, so nobody reads this document as a description of
production behaviour.

Wiring it changes no score and no threshold by construction — the function
returns the risk and tier unchanged — so the integration is additive: a
confidence field and an adjustment log alongside an unchanged decision.

---

## 6. Limits

- **64 positives.** Every figure here rests on the same small positive class as
  the main model. Zero mules in a 1,452-account band is consistent with a good
  separator and also with there simply not being many mules to place.
- **Single repeat.** The OOF PR-AUC is one repeat; the tournament's
  three-repeat evidence on a narrower version of this view shows a ± 0.08
  spread. Treat 0.64455 accordingly.
- **In-sample bands.** See §4.2.
- **The verifier is not a merchant classifier.** It never asserts that an
  account is a merchant. It reports how far the business evidence fails to
  corroborate the alert, which is a different and weaker claim.
- **Strong business evidence is not a clearance.** A band label of
  `STRONG_BUSINESS_EVIDENCE` means "not currently flagged by the business
  evidence"; it never means an account is safe, clean or cleared, and it never
  removes an account from review.
