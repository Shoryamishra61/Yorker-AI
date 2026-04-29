# IPL Match Prediction Research Project
## Research Product Requirements Document

**Version:** 2.0.0  
**Date:** April 24, 2026  
**Status:** Research Planning  
**Classification:** Internal - Research, Data Science, Engineering  

---

## 1. Purpose

The current IPL winner model is useful as a prototype, but it is not yet credible enough to present as a finished public prediction engine. It trained and produced outputs quickly because the first version relied mostly on readily available Cricsheet match data, engineered team/player aggregates, and a small validation window.

This document reframes the project as a proper research program. The goal is to build evidence gradually: first data quality, then features, then model comparison, then uncertainty, then public storytelling.

The public output should become a research series, not a single Twitter post.

---

## 2. Research Thesis

IPL match prediction can be improved by moving from simple team-level history toward match-context modeling:

- player availability and expected XI strength
- batting and bowling phase matchups
- venue-specific scoring and dismissal patterns
- toss, dew, weather, and pitch interactions
- recent form with regression toward long-term ability
- squad construction and auction-driven role changes
- uncertainty from small samples and lineup ambiguity

The project should prove which signals actually add predictive value. A feature is included only if it improves backtested performance, calibration, interpretability, or research insight.

---

## 3. Current State

The repository already contains a working MVP:

- Cricsheet IPL JSON ingestion
- processed match and feature tables
- a recency-weighted logistic model
- benchmark artifacts
- upcoming fixture predictions
- Twitter draft output

However, current results are not strong enough for confident public claims:

| Metric / Check | Current Observation |
|---|---:|
| 2025 strict pre-match holdout accuracy | 67.1% on 70 matches |
| 2025 strict pre-match Brier score | 0.230 |
| 2025 strict pre-match ROC-AUC | 0.704 |
| Strict pre-match rolling weighted accuracy | 53.5% across 11 validation seasons |
| Toss-confirmed XI rolling weighted accuracy | 55.8% across 11 validation seasons |
| Best single-feature baseline rolling accuracy | 51.4% |

Interpretation: the MVP may have found some signal, but the result is not robust enough across seasons. The next phase must prioritize data provenance, leakage checks, broader backtesting, and feature ablations.

---

## 4. Success Criteria

This project should not optimize for one impressive number. It should optimize for credible research.

### Research Success

| Goal | Acceptance Criteria |
|---|---|
| Reproducible data | Every source has provenance, schema, refresh cadence, and license notes |
| Leakage-safe evaluation | Features available only before match start are used for prediction |
| Robust backtesting | Walk-forward validation over multiple seasons |
| Baseline comparison | Model beats naive, Elo, recent-form, venue, and market-implied baselines where available |
| Calibration | Brier score and reliability curves reported, not only accuracy |
| Explainability | Top factors shown with caveats and stability checks |
| Public honesty | Posts distinguish facts, model outputs, assumptions, and uncertainty |

### Model Targets

These are research targets, not claims:

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|---|---:|---:|---:|
| Walk-forward accuracy | > 56% | > 60% | > 63% |
| Brier score | < 0.250 | < 0.235 | < 0.220 |
| ROC-AUC | > 0.56 | > 0.62 | > 0.66 |
| Calibration ECE | < 0.12 | < 0.09 | < 0.07 |
| Baseline lift | positive | statistically meaningful | stable across seasons |

No public claim should imply certainty beyond these measured results.

---

## 5. Project Phases

### Phase 0 - Audit the MVP

Objective: understand what the current model really learned.

Tasks:

- inventory all generated datasets and artifacts
- document current feature columns and their creation time
- identify leakage risks, especially post-match information in pre-match features
- compare one-season holdout against rolling walk-forward performance
- inspect failure cases by season, venue, team, toss, and margin
- remove or rewrite public copy that overstates the model

Deliverables:

- `docs/research/00_mvp_audit.md`
- leakage checklist
- metric table with confidence intervals
- revised Twitter draft labeled as research log, not launch post

Exit criteria:

- every current metric is reproducible from a command
- no unsupported claim remains in public-facing copy

---

### Phase 1 - Data Collection and Provenance

Objective: build a trusted data layer before adding complex models.

Data sources to collect:

| Dataset | Primary Source | Purpose | Priority |
|---|---|---|---|
| Ball-by-ball history | Cricsheet IPL JSON | deliveries, wickets, phases, player participation | P0 |
| Match metadata | Cricsheet + official IPL fixtures | date, venue, teams, toss, result | P0 |
| Player registry | Cricsheet people registry / IPL squads | stable player identifiers | P0 |
| Squads and auction | official IPL/BCCI/team announcements | role, price, retention, transfers | P1 |
| Playing XI | scorecards / official match center | availability and lineup strength | P1 |
| Venue metadata | historical scorecards + stadium info | venue priors and conditions | P1 |
| Weather | weather API or archived observations | humidity, dew proxy, rain risk | P2 |
| Pitch reports | official broadcast/media text | pitch labels and curator notes | P2 |
| Odds / market baseline | legal historical odds source if available | benchmark against crowd wisdom | P3 |

Data quality checks:

- row counts by season and match
- missing player IDs
- duplicate matches
- team-name normalization
- venue-name normalization
- innings completeness
- abandoned/no-result handling
- schema versioning

Deliverables:

- `data_catalog.md`
- source-specific ingestion scripts
- raw-to-processed lineage table
- data quality report generated on every run

Exit criteria:

- the project can explain exactly where each model field came from
- all P0 datasets are reproducible locally

---

### Phase 2 - Feature Research

Objective: create candidate features, then prove which ones matter.

Feature families:

| Family | Examples | Risk |
|---|---|---|
| Team strength | Elo, recent win rate, NRR trend, squad stability | can overfit recent streaks |
| Batting | rolling runs, strike rate, balls faced, phase scoring, venue splits | sparse for new players |
| Bowling | economy, wickets, dot-ball rate, death-over economy, phase usage | role changes across teams |
| Matchups | batter-vs-bowler balls, dismissal rate, strike rate | very sparse and noisy |
| Venue | average score, chase rate, boundary size proxy, spin/pace bias | venue changes over time |
| Conditions | toss, innings, dew proxy, temperature, rain | often unavailable pre-match |
| Availability | confirmed XI, predicted XI, overseas balance, injuries | source reliability issues |
| Auction/squad | price, retention, role scarcity, new-team adjustment | price is not always performance |

Feature rules:

- every feature gets a definition, availability time, missing-value policy, and expected direction
- pre-match models cannot use toss or confirmed XI unless running the match-day version
- feature importance must be checked with ablation, not only model importance
- sparse player matchup features must shrink toward role/team priors

Deliverables:

- `feature_catalog.md`
- feature generation tests
- feature ablation report
- leakage-safe `match_features_prematch.csv`
- match-day `match_features_toss_xi.csv`

Exit criteria:

- features are grouped by prediction time: pre-season, match-eve, toss-confirmed
- each feature family has measured lift or is removed/deprioritized

---

### Phase 3 - Modeling Experiments

Objective: compare simple and complex models under the same validation protocol.

Candidate models:

- naive home/team prior
- recent-form baseline
- Elo and Glicko-style ratings
- regularized logistic regression
- random forest
- gradient boosting
- calibrated gradient boosting
- hierarchical Bayesian model for player/team shrinkage
- ensemble only if individual models justify it

Validation protocol:

- walk-forward validation by season
- train only on matches before the validation match date
- report per-season and aggregate metrics
- evaluate calibration with reliability curves
- bootstrap confidence intervals
- compare against baselines with paired tests

Deliverables:

- experiment registry
- benchmark table by season
- calibration plots
- failure-case notebook/report
- chosen model card

Exit criteria:

- selected model beats simple baselines consistently enough to justify added complexity
- probabilities are calibrated well enough for public reporting

---

### Phase 4 - Scenario and Uncertainty Modeling

Objective: stop presenting predictions as single-point truth.

Required outputs:

- win probability with confidence band
- lineup uncertainty scenarios
- toss scenario split: bat first vs chase
- venue/weather sensitivity
- top drivers and counterfactuals
- "what would flip the prediction" explanation

Example:

```text
RCB vs MI
Pre-toss estimate: RCB 54% [48%, 60%]
If RCB chase: 58%
If MI chase: 52%
Biggest swing factor: confirmed death bowling combination
```

Deliverables:

- scenario simulation module
- uncertainty report format
- prediction card template
- assumptions log for every published prediction

Exit criteria:

- every public prediction includes uncertainty and assumptions

---

### Phase 5 - Research Publication Series

Objective: turn the project into a credible public research narrative.

Instead of one Twitter post, publish a phased series:

1. Data collection: what IPL data is available and what is missing
2. Data cleaning: player IDs, team names, venues, no-results, super overs
3. Baselines: how hard is IPL prediction before ML?
4. Feature research: which cricket signals actually help?
5. Model comparison: simple models vs boosted models
6. Calibration: why 60% should mean 60%
7. Case studies: matches the model got right and wrong
8. Live prediction protocol: how match-eve predictions are generated
9. Final season review: honest scorecard after results are known

Each post should include:

- one concrete chart or table
- one reproducible command or artifact
- one limitation
- no exaggerated certainty

Deliverables:

- `reports/research_posts/`
- charts for each phase
- public metric dashboard
- final research README

Exit criteria:

- public story matches measured evidence
- claims are understandable to both cricket fans and ML practitioners

---

## 6. Engineering Workstreams

### Data Engineering

- source adapters
- schema normalization
- data quality checks
- reproducible raw data snapshots
- feature store tables by prediction time

### Modeling

- leakage-safe feature pipeline
- baseline models
- model experiments
- calibration
- uncertainty estimation
- error analysis

### Reporting

- research reports
- charts
- Twitter/X thread drafts
- model cards
- prediction cards
- dashboard updates

### Product/Demo

- local dashboard
- API or CLI prediction endpoint
- match-eve prediction workflow
- artifact versioning

---

## 7. Repository Structure Target

```text
data/
  raw/
  interim/
  processed/
docs/
  research/
    00_mvp_audit.md
    01_data_catalog.md
    02_feature_catalog.md
    03_benchmark_report.md
    04_model_card.md
reports/
  charts/
  research_posts/
  twitter/
src/
  ipl_ml/
    data/
    features/
    models/
    evaluation/
    reporting/
artifacts/
  models/
  evaluations/
  data_quality/
```

The repo does not need to be reorganized immediately, but new work should move toward this structure.

---

## 8. Public Claim Policy

Allowed:

- "Strict pre-match prototype result on 2025 holdout: 67.1% accuracy over 70 matches."
- "Rolling validation is much lower, so this is still research."
- "The next step is data and feature validation."
- "The model currently beats some baselines in one validation window but not robustly across all seasons."

Not allowed:

- "Best-in-class for cricket" unless externally benchmarked
- "97% accuracy" unless achieved out-of-sample with a documented protocol
- "Highest-signal feature" unless proven by ablation
- live player/points-table claims without dated, sourced verification
- title predictions framed as confident outcomes

---

## 9. Immediate Next Sprint

### Sprint Goal

Convert the project from "quick model + Twitter post" into "research-ready IPL prediction study."

Tasks:

1. Write MVP audit from current artifacts. (started)
2. Build a data catalog for Cricsheet and official schedule feeds. (started)
3. Add leakage checklist to feature generation. (started)
4. Split features into pre-match, toss-confirmed, and post-match-only.
5. Add ablation experiments for feature families.
6. Replace Twitter thread with research-series post 1.
7. Update README to communicate research status.

Acceptance criteria:

- README no longer implies final model readiness
- PRD describes phased research plan
- Twitter draft no longer presents predictions as a launch claim
- benchmark report includes rolling-season results and limitations

---

## 10. Final Deliverable Vision

The finished project should look like a serious sports analytics research repository:

- transparent data sourcing
- reproducible experiments
- clear baselines
- credible uncertainty
- interpretable match predictions
- honest post-match review
- a public series that builds trust over time

The project should earn attention because the research process is visible, not because one post claims a big number.
