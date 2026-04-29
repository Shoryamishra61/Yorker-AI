# Benchmark Report

This report is generated from leakage-aware walk-forward validation.

## Quality Gate

- Candidate: `recency_l1_logistic__prematch_core`
- Validation season: 2025
- Target accuracy: 0.630
- Observed accuracy: 0.671
- 95% CI: [0.594, 0.749]
- Passed: true

## Candidate Summary

| Candidate | Feature Set | Kind | Seasons | Weighted Accuracy | Fixture Accuracy | ROC-AUC | Brier | Log Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| recency_l1_logistic__toss_confirmed_xi | toss_confirmed_xi | model | 11 | 0.558 | 0.558 | 0.556 | 0.252 | 0.698 |
| regularized_logistic__prematch_core | prematch_core | model | 11 | 0.547 | 0.548 | 0.528 | 0.258 | 0.712 |
| regularized_logistic__toss_confirmed_xi | toss_confirmed_xi | model | 11 | 0.542 | 0.542 | 0.550 | 0.260 | 0.718 |
| regularized_logistic_with_identity__toss_confirmed_xi | toss_confirmed_xi | model | 11 | 0.536 | 0.536 | 0.554 | 0.257 | 0.712 |
| recency_l1_logistic__prematch_core | prematch_core | model | 11 | 0.535 | 0.536 | 0.539 | 0.252 | 0.698 |
| regularized_logistic_with_identity__prematch_core | prematch_core | model | 11 | 0.529 | 0.530 | 0.535 | 0.257 | 0.710 |
| shallow_random_forest__prematch_core | prematch_core | model | 11 | 0.529 | 0.525 | 0.527 | 0.252 | 0.697 |
| shallow_random_forest__toss_confirmed_xi | toss_confirmed_xi | model | 11 | 0.523 | 0.531 | 0.537 | 0.251 | 0.696 |
| baseline_recent_win_rate_diff | single_feature_baseline | baseline | 11 | 0.514 | 0.511 | 0.518 | 0.251 | 0.695 |
| baseline_season_win_rate_diff | single_feature_baseline | baseline | 11 | 0.505 | 0.514 | 0.510 | 0.251 | 0.696 |
| baseline_venue_win_rate_diff | single_feature_baseline | baseline | 11 | 0.503 | 0.517 | 0.503 | 0.252 | 0.698 |
| baseline_elo_expected_team_a | single_feature_baseline | baseline | 11 | 0.502 | 0.502 | 0.508 | 0.261 | 0.717 |
| baseline_elo_home_expected_team_a | single_feature_baseline | baseline | 11 | 0.500 | 0.501 | 0.508 | 0.262 | 0.719 |

## Credibility Notes

- Best model by weighted accuracy: `recency_l1_logistic__toss_confirmed_xi` at 0.558.
- Best baseline by weighted accuracy: `baseline_recent_win_rate_diff` at 0.514.
- Model lift over best baseline: +0.044.
- `prematch_core` excludes confirmed-XI player aggregate features.
- `toss_confirmed_xi` includes features that are only defensible after teams are known.
- Public claims should use walk-forward metrics, not only a single validation season.

## Per-Season Primary Candidate

| Season | Accuracy | Fixture Accuracy | ROC-AUC | Brier | Validation Matches |
|---:|---:|---:|---:|---:|---:|
| 2016 | 0.550 | 0.550 | 0.486 | 0.268 | 60 |
| 2017 | 0.552 | 0.552 | 0.628 | 0.243 | 58 |
| 2018 | 0.583 | 0.583 | 0.633 | 0.244 | 60 |
| 2019 | 0.544 | 0.544 | 0.580 | 0.244 | 57 |
| 2020 | 0.411 | 0.411 | 0.364 | 0.261 | 56 |
| 2021 | 0.492 | 0.492 | 0.505 | 0.253 | 59 |
| 2022 | 0.480 | 0.486 | 0.500 | 0.260 | 74 |
| 2023 | 0.589 | 0.589 | 0.547 | 0.256 | 73 |
| 2024 | 0.493 | 0.493 | 0.514 | 0.253 | 71 |
| 2025 | 0.671 | 0.671 | 0.704 | 0.230 | 70 |
| 2026 | 0.435 | 0.435 | 0.473 | 0.260 | 23 |
