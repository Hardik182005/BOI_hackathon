"""Post-model USP layers and the machinery that proves they changed nothing.

Two products live under this package's remit:

* **Cohort Radar** (``muleguard.cohort``) - behavioural-similarity retrieval
  across the portfolio.
* **Account-Control Ambiguity** (``muleguard.explain.control_attribution``) -
  the statement of what a behavioural score does not establish.

Neither may alter a risk probability, a tier or a policy action. This package
holds the evidence for that claim rather than the claim itself: an invariant
snapshot taken before the upgrade, the same snapshot taken after, and the
comparison between them.
"""
