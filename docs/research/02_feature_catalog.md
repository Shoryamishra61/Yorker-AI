# Feature Catalog

**Date:** April 24, 2026  
**Status:** Draft backed by `src/ipl_ml/feature_registry.py`  

## Feature Sets

| Feature Set | Intended Use | Includes | Excludes |
|---|---|---|---|
| `prematch_core` | Match-eve prediction before toss and confirmed XI | schedule, venue priors, Elo, team form, head-to-head, rest | confirmed-XI player aggregates, target/result fields |
| `toss_confirmed_xi` | Prediction after official team sheet is known | all `prematch_core` fields plus confirmed-XI player aggregates | target/result fields |

The persisted model currently uses `prematch_core` numeric features only. This prevents the model from learning from confirmed playing XIs when predicting fixtures where the XI is not yet known.

## Current Selected Features

See `artifacts/models/feature_columns.json` for the exact selected list. Current selected families:

- schedule: season, month, playoff flag
- team strength: Elo and historical win-rate differences
- current form: season win rate, NRR proxy, runs for/against, streaks
- recent form: last-five win rate, NRR proxy, scoring and conceded runs
- phase form: recent powerplay runs and wickets
- venue: venue win rate, home advantage, scoring and bowling-style priors
- head-to-head: historical team-vs-team win-rate difference
- rest: days since last match

## Leakage Controls

Generated audit:

```text
artifacts/models/feature_availability_audit.csv
```

The audit marks each column by:

- feature set
- availability time
- feature family
- whether the selected model actually uses it
- description

Rules:

- `label` is post-match only and must never be a feature.
- confirmed-XI features are excluded from `prematch_core`.
- toss and confirmed XI can only be used in a separate `toss_confirmed_xi` model.
- post-match score, margin, wickets, and result fields are forbidden for pre-match modeling.

## Next Work

1. Add tests that fail if post-match fields enter a pre-match feature set.
2. Add feature-family ablation reports.
3. Build a true likely-XI dataset so player-level features can be used before toss without training on confirmed XI.
4. Add weather and pitch features with source timestamps.
