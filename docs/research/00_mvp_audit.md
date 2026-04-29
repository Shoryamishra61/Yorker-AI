# MVP Audit

**Date:** April 24, 2026  
**Status:** Initial audit from existing artifacts  

## Summary

The current repository has a working IPL prediction MVP, but the evidence does not yet support presenting it as a finished prediction engine. The strongest metric is the 2025 strict pre-match holdout result, while broader rolling validation is much weaker.

## Existing Pipeline

Observed capabilities:

- Cricsheet IPL JSON ingestion
- official schedule ingestion
- processed match table generation
- match feature generation
- recency-weighted logistic model
- benchmark comparison
- upcoming fixture prediction
- Twitter/report artifact generation

Primary commands:

```bash
PYTHONPATH=src python3 -m ipl_ml.cli run-all
PYTHONPATH=src python3 -m ipl_ml.cli benchmark --target-accuracy 0.97
```

## Current Metrics

From `artifacts/models/evaluation_metrics.json`:

| Metric | Value |
|---|---:|
| Validation season | 2025 |
| Validation matches | 70 |
| Feature set | prematch_core |
| Accuracy | 0.671 |
| Balanced accuracy | 0.671 |
| Brier score | 0.230 |
| Log loss | 0.653 |
| ROC-AUC | 0.704 |
| Expected calibration error | 0.114 |

From `artifacts/models/benchmark_metrics.json`:

| Check | Value |
|---|---:|
| Target accuracy gate | 0.630 |
| Observed 2025 strict pre-match accuracy | 0.671 |
| Gate passed | true |
| Strict pre-match rolling weighted accuracy | 0.535 |
| Toss-confirmed XI rolling weighted accuracy | 0.558 |
| Best single-feature baseline rolling accuracy | 0.514 |

## Interpretation

The 2025 holdout result may indicate useful signal, but it is not enough on its own. Rolling-season validation suggests the model is only modestly above simple baselines and may be unstable across seasons.

The most likely causes are:

- small validation windows
- feature instability across seasons
- player and team composition changes
- possible leakage or availability-time ambiguity
- over-reliance on recent form
- incomplete weather, pitch, and XI data

## Leakage Risks To Review

Features must be classified by when they become available:

| Feature Type | Pre-season | Match-eve | Toss-confirmed | Post-match only |
|---|---:|---:|---:|---:|
| historical team records | yes | yes | yes | no |
| rolling player form before match | yes | yes | yes | no |
| official fixture venue | yes | yes | yes | no |
| weather forecast | no | yes | yes | no |
| toss winner/decision | no | no | yes | no |
| confirmed playing XI | no | no | yes | no |
| innings score | no | no | no | yes |
| result/margin | no | no | no | yes |

Any feature derived from innings score, wickets, result, final NRR, or post-match player stats must be excluded from pre-match training rows.

## Immediate Actions

1. Generate a complete feature catalog.
2. Mark every feature with availability time.
3. Add automated checks for post-match-only fields.
4. Re-run walk-forward validation with only pre-match-safe features.
5. Report baselines and model metrics side by side.
6. Replace launch-style social copy with research log copy.

## Public Claim Guidance

Safe wording:

- "The strict pre-match prototype reached 67.1% on a 2025 holdout, but rolling validation is much lower."
- "The project is still in research."
- "Next work: source documentation, leakage checks, feature ablation, and calibration."

Unsafe wording:

- "Best-in-class"
- "Elite accuracy"
- "Ready for live betting or public picks"
- "Highest-signal feature" without ablation evidence
- title predictions presented as confident outcomes
