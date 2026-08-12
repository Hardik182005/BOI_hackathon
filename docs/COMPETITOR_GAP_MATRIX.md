# Competitor and Industry Gap Matrix

Master prompt §4, addendum UPDATE 7.

**Method and its limits.** Each repository below was inspected **read-only** on
2026-08-12 via its public GitHub landing page — README text and the visible
top-level file listing. Nothing was cloned, no source file was read in full, and
**no competitor code, asset or design was copied**. Where a row describes what a
repository *contains*, that is verified from the file listing. Where a row
describes a *claim*, it is the authors' own README claim, reported as a claim and
labelled as such. A number in a README that cannot be reproduced from the
repository is not evidence, and this document never treats it as one.

The goal is **not** "more features than everybody". It is a system that is more
defensible under a hidden validation set and more useful to the analyst who has
to act on it.

---

## 1. Repositories inspected

| # | Repository | Status | What it actually contains | Same dataset as ours? |
|---|---|---|---|---|
| C1 | `vishwa-11web/mule-detection-iit-hyderabad-bank-of-india-hackathon` | **VERIFIED** | README + 16 Python modules — the "ASVFS Pipeline" | Not stated |
| C2 | `Durgesh-18/Mule-Account-…-Temporal-Activity-Window-Detection` | **VERIFIED** | README + 2 Jupyter notebooks | **No** — RBIH challenge, 7.4 M transactions |
| C3 | `thekarak/Muleeye-BOI-IITH-2026` | **VERIFIED** | **README.md only — no source code in the repository** | **Yes** — cites 9,082 accounts, 0.89 % mule rate |
| C4 | `shaikharman8814-cloud/money-muling-detection` | **VERIFIED** | Flask `app.py`, static/templates, sample CSV | No — edge-list CSV input |
| C5 | `Samrudhp/MM_Detection` | **VERIFIED** | FastAPI backend + React/Cytoscape frontend, deploy configs | No — edge-list CSV input |
| C6 | `rupeshbharambe24/National-Fraud-Prevention-Challenge-Phase-1` | **VERIFIED** | One EDA notebook + CSVs; **Phase 1, explicitly no model** | **No** — RBIH, ~40 K accounts |
| C7 | `dikshaashadeep/fraudlens-hackathon` | **VERIFIED** | Streamlit `app.py`, synthetic datasets | No — synthetic transactions |

Industry references (BioCatch, LexisNexis, FacePhi, Feedzai, Bureau, Innefu, ISB
research, academic LightGBM+SHAP+LLM work, temporal-graph research) were used as
**design context only**. They are commercial products without public source, so
no capability claim about them is made here — `NOT_VERIFIED` by construction.

### The most important structural finding

**Only C3 is on our dataset.** C2 and C6 are the RBIH National Financial Products
Challenge — 7.4 M transactions, ~40 K accounts, **a real transaction table with
counterparties**. C4, C5 and C7 take an uploaded edge list or synthetic data.

That single fact reframes the entire comparison. **A solution built on a
transaction table is solving a different problem from one built on 9,082 rows of
pre-aggregated account features**, and its metrics are not comparable to ours in
either direction. C2's headline AUC-ROC of 0.999833 is on data that contains
7.4 M individual transactions and a 3.25 M-edge graph. We have neither. Quoting
that number against ours — in our favour or theirs — would be dishonest.

---

## 2. The matrix

Legend: **●** present and verifiable from the repository · **◐** claimed in the
README, source not present or not read · **○** not present · **n/a** not
applicable to that repo's problem shape.

| Dimension | **MuleGuard · Trinetra** | C1 ASVFS | C2 RBIH ensemble | C3 Muleeye | C4 Flask demo | C5 MM_Detection | C6 EDA only | C7 FraudLens |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Classifier** | ● XGBoost on 120 stability-selected features; 5×3 repeated stratified OOF | ◐ GBM + ODE net | ◐ 6-model ensemble, Borda + Dirichlet blend | ◐ XGB+SMOTE / IF / rules, α.60 β.25 γ.15 | ○ rules only | ○ rules only | ○ none | ◐ IsolationForest + rules |
| **Graph** | ● adapter built, reports `UNAVAILABLE` — refuses to fabricate edges | ◐ GraphSAGE, Louvain, persistent homology | ◐ 3.25 M-edge graph, 1/2/3-hop contamination | ◐ NetworkX + force-graph | ● real graph from uploaded edges | ● real graph from uploaded edges | ○ | ● NetworkX on synthetic edges |
| **Explainability** | ● **dual-evidence ProofGraph** — prosecution *and* defence, every node sourced | ◐ TreeSHAP into a feedback loop | ○ feature importance only | ◐ SHAP waterfall cards | ○ rule text | ● itemised rule breakdown | ○ | ◐ Gemini API narrative |
| **False-positive control** | ● 5 layers: no-accusation language, defence panel, learned merchant verifier, conformal abstention, calibration | ○ | ○ | ○ | ○ | ◐ "merchant dampening rule" | ○ | ○ |
| **Validation handling** | ● Sealed Validation Protocol — predictions hashed **before** labels are readable | ○ | ● adversarial validation, 2-stage | ◐ single 80/20 split | ○ | ○ schema check | ○ | ○ |
| **Leakage protection** | ● Feature Availability Firewall; 13 columns quarantined pre-decision | ○ | ● 10 of 206 features excluded by adversarial AUC | ○ | n/a | n/a | ○ | n/a |
| **Hidden-validation robustness** | ● seed-noise floor, 15-round positive-removal stress test, published **LOW** badge | ○ | ◐ CV only | ○ | ○ | ○ | ○ | ○ |
| **Case-management UX** | ● full analyst workspace: queue, case view, ProofGraph, Validation Lab, Graph Lab | ○ | ○ | ◐ described, no code | ● single page | ● React app | ○ notebook | ● Streamlit dashboard |
| **Deployment** | ● offline, one command, no API keys, no internet | ○ | ○ notebooks | ○ | ● local Flask | ● Render + Vercel | ○ | ◐ local Streamlit, **needs Gemini API key** |
| **LLM role** | ● explains verified evidence only; scoring runs with Ollama stopped | ○ | ○ | ○ | ○ | ○ | ○ | ◐ **LLM generates the investigation summary** |

---

## 3. Where competitors are genuinely ahead of us

Stating this first, because a gap matrix that only flatters its author is
worthless.

| They have | Who | Our position |
|---|---|---|
| **A real transaction graph with millions of edges** | C2, C6 | We have none, because our extract has none. C2's 1/2/3-hop contamination features are genuinely powerful and we cannot build them. Our answer is the Graph Adapter: the moment an edge file exists, it works — but today it reports `UNAVAILABLE` |
| **Adversarial validation used to *drop* features** | C2 | We run an adversarial shield but it **warns only** and never changes a prediction (UPDATE 3). C2 excluding 10 of 206 features on train/test AUC is a legitimate technique we chose not to adopt, because our "test" is hidden and dropping features against a proxy risks discarding real signal |
| **A deployed public URL** | C5 | Ours runs offline by design — no external host, no API key. That is a deliberate trade: judge-reproducibility over convenience |
| **Transaction-level temporal windows** | C2 | Our data is pre-aggregated into L7D/L14D/L31D buckets. Their temporal IoU of 0.6879 addresses a sub-problem our data cannot express |

---

## 4. The comparison that matters: C3, the one on our data

C3 (Muleeye) is the only inspected repository working on the same 9,082-account,
0.89 %-prevalence extract. Its README reports **PR-AUC 0.86, recall 0.91,
precision 0.78**. Ours reports OOF PR-AUC 0.769 ± 0.027 and locked-test 0.7263.

Taken at face value, they beat us. Four verified facts about that comparison:

1. **The repository contains only `README.md`.** No pipeline, no model, no
   notebook, no metrics file. This is verified from the public file listing, not
   inferred. The number cannot be reproduced by anyone, including a judge.
2. **It is a single stratified 80/20 split.** With 81 positives, the test fold
   holds ~16. PR-AUC on 16 positives has enormous variance — our own measured
   **seed-noise floor is 0.0905 PR-AUC**, meaning differences smaller than that
   on this data are not differences at all. A single-split 0.86 and a
   repeated-OOF 0.77 are not separated by any margin this dataset can resolve.
3. **No leakage control is described.** This is decisive here. We measured it:
   admitting the single post-resolution column `F3912` moves PR-AUC from
   **0.6142 to 0.9419**. On this dataset, a high headline number is *more likely*
   to be evidence of a quarantine failure than of modelling skill. We cannot
   claim C3 leaked — there is no code to inspect, so the honest verdict is
   **`NOT_VERIFIABLE`** — but any team reporting a strong number here owes the
   judges a leakage audit, and ours is `docs/LEAKAGE_AUDIT.md` and
   `docs/FEATURE_AVAILABILITY_AUDIT.md`.
4. **SMOTE on 81 positives.** C3 reports SMOTE lifting recall 61 % → 88 %. We
   tested resampling and rejected it: synthesising minority points by
   interpolating between 81 real mules in a 3,924-dimensional space manufactures
   examples that correspond to no real account, and it inflates cross-validated
   recall while degrading calibration — which is why our probabilities carry
   Brier 0.0031 and ECE 0.0015.

**The gap we claim over C3 is not accuracy. It is verifiability.** Our number
comes with a sealed prediction hash timestamped before any label was readable, a
locked test set with a touch log, and a repeated-OOF estimate with its noise floor
published.

---

## 5. What we deliberately refused to copy (UPDATE 7)

C1's ASVFS pipeline is the reference case, and the addendum names it directly.
Verified from its file listing: `hybrid_ode_solver.py`, `topological_ml.py`,
`few_shot_network.py`, `trajectory_generation.py`, `loop2_shield.py`,
`label_propagation.py`.

| Their technique | Why we did not build it |
|---|---|
| ODE-network risk scoring | 81 positives. A continuous-time dynamical model has no signal to fit here, and no analyst can act on its output |
| GraphSAGE embeddings | there is no graph. A GNN over fabricated edges is fabrication with a neural network on top |
| Persistent homology | the same, with a harder-to-question output |
| Synthetic trajectory generation | invented data presented as evidence |
| Contextual bandits / auto-adapting thresholds | a threshold that moves on its own cannot be audited after an incident. Ours are frozen at `policy_version 1.0` |
| Few-shot prototype networks | 81 positives across an unknown number of typologies |
| Automatic fund freezing | no such code path exists in our system; every high-impact action requires human approval |

And from C2, verified from its README: a **Dirichlet random-blend search over
model weights**. UPDATE 2 forbids it — *"Do NOT add a complex 5,000-random-blend
search"*. A blend weight tuned by random search over folds containing ~16
positives fits fold noise. We use rank-stable ensembling with fixed weights
instead.

**Every one of these would have made a better slide.** None of them would have
survived a judge asking "how do you know that helped?"

---

## 6. What no inspected repository has

These are gaps in the field, not merely gaps against one competitor. Each is
verified as absent from all seven repositories' public READMEs and file listings.

1. **A defence case.** Every system above answers "why is this suspicious?" Not
   one answers "why might this be wrong?" The ProofGraph's defence column and its
   six measured structural-doubt triggers are, on this evidence, unique — and
   they are the direct mechanism for the false-positive problem that dominates a
   0.89 %-prevalence queue.
2. **A cryptographic seal on the honesty of the evaluation.** C2 has strong
   validation discipline; nobody has an artefact proving predictions predated the
   labels.
3. **A published failure.** We publish a robustness badge of **LOW**, driven by
   `prediction_rank_stability` 0.3694, against thresholds fixed before the
   measurement. No inspected repository publishes a metric that makes it look
   worse.
4. **A stated list of what the data cannot support.**
   `PATTERN_AVAILABILITY_MATRIX.md` marks 6 of 22 typologies `NOT_AVAILABLE` —
   including the four most demo-friendly ones.
5. **An LLM that is structurally barred from scoring.** C7 has the honest
   opposite arrangement: Gemini writes the investigation summary, which means the
   explanation is generated text rather than derived evidence, and the system
   needs an API key and internet. Ours explains only nodes that already exist,
   and core scoring runs with Ollama stopped.

---

## 7. Honest summary

| | |
|---|---|
| Where we are behind | no real transaction graph (C2 has one; our data does not) |
| Where we are level | headline discrimination — and on this dataset the differences are inside the noise floor |
| Where we are ahead | leakage control, evaluation integrity, false-positive protection, dual-evidence explanation, offline reproducibility |
| What we refused | every technique in UPDATE 7 — each of which would have demoed better and defended worse |

The defensible claim is narrow and we will make only that one: **among the
systems we could verify, MuleGuard is the one whose reported number a judge can
check.**
