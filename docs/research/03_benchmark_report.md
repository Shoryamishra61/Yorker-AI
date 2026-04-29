# Benchmark Report

**Date:** April 24, 2026  
**Status:** Summary of generated benchmark artifacts  

The generated benchmark report lives at:

```text
artifacts/models/benchmark_report.md
```

## Current Result Summary

| Candidate | Feature Set | Rolling Weighted Accuracy | ROC-AUC | Brier |
|---|---|---:|---:|---:|
| recency L1 logistic | toss-confirmed XI | 0.558 | 0.556 | 0.252 |
| regularized logistic | pre-match core | 0.547 | 0.528 | 0.258 |
| recency L1 logistic | pre-match core | 0.535 | 0.539 | 0.252 |
| best single-feature baseline | recent win-rate diff | 0.514 | 0.518 | 0.251 |

## Interpretation

The confirmed-XI feature set performs best, but it is only valid after team sheets are known. The stricter pre-match model is more credible for match-eve prediction, but its rolling accuracy is only modestly above baseline.

The 2025 strict pre-match holdout result is stronger:

| Metric | Value |
|---|---:|
| Accuracy | 0.671 |
| 95% CI | 0.594-0.749 |
| ROC-AUC | 0.704 |
| Brier score | 0.230 |
| ECE | 0.114 |

This should be described as a promising validation window, not a final model result.

## Accuracy Improvement Notes

A categorical identity experiment using team and venue one-hot features did not improve rolling accuracy. It remains in the benchmark as evidence, but it should not become the selected model until it improves walk-forward results.

The most credible path to improved accuracy is better data, not more model complexity:

- true likely-XI history
- player role and batting position data
- injuries and availability
- pitch/weather source timestamps
- feature-family ablation
- calibration by season and confidence bucket
