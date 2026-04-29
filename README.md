# IPL Match Prediction Research

This repo is being developed as a research-grade IPL match prediction project.

The first MVP already runs end to end, but the current evidence is mixed: the stricter pre-match model reports 67.1% accuracy on the 2025 holdout over 70 matches, while rolling validation across seasons is much closer to baseline performance. The project is therefore moving into a phased research workflow focused on data quality, feature validation, leakage-safe evaluation, calibration, and honest public reporting.

## Current Capabilities

- Cricsheet IPL JSON ingestion
- official schedule feed ingestion
- match and feature table generation
- baseline and model benchmarking
- feature availability and leakage audit
- data quality reports
- upcoming fixture scoring
- local dashboard artifacts
- draft reporting outputs

## Research Phases

1. MVP audit and leakage review
2. Data collection and provenance catalog
3. Feature extraction and ablation studies
4. Model comparison with walk-forward validation
5. Scenario and uncertainty modeling
6. Public research post series
7. Final season review

See [PRD.md](/Users/admin67/IPL_ML/PRD.md) for the full research plan.

## Quick Start

Run the current full pipeline:

```bash
PYTHONPATH=src python3 -m ipl_ml.cli run-all
```

Useful commands:

```bash
PYTHONPATH=src python3 -m ipl_ml.cli download-data
PYTHONPATH=src python3 -m ipl_ml.cli build-dataset
PYTHONPATH=src python3 -m ipl_ml.cli train
PYTHONPATH=src python3 -m ipl_ml.cli benchmark --target-accuracy 0.97
PYTHONPATH=src python3 -m ipl_ml.cli predict-upcoming
PYTHONPATH=src python3 -m ipl_ml.cli report
```

## Important Artifacts

- Processed dataset: `data/processed/match_features.csv`
- Evaluation metrics: `artifacts/models/evaluation_metrics.json`
- Rolling benchmark metrics: `artifacts/models/benchmark_metrics.json`
- Season benchmark table: `artifacts/models/benchmark_by_season.csv`
- Validation predictions: `artifacts/models/validation_predictions.csv`
- Upcoming predictions: `artifacts/models/upcoming_predictions.csv`
- Research/public draft: `reports/twitter/thread.md`
- Feature availability audit: `artifacts/models/feature_availability_audit.csv`
- Benchmark report: `artifacts/models/benchmark_report.md`
- Data quality summary: `artifacts/data_quality/data_quality_summary.json`

## Current Research Caveat

Do not treat the current model as a finished predictor. Before public launch, the project needs stronger source documentation, feature ablations, leakage checks, uncertainty reporting, and multi-season validation.
