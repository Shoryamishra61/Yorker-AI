# Yorker AI 🏏

**ML-powered IPL 2026 match winner predictions** — trained on 1,100+ real matches, walk-forward validated across 11 seasons.

> **Live Dashboard:** [huggingface.co/spaces/Shoryamishra61/yorker-ai](https://huggingface.co/spaces/silentkiller61/Yorker-AI)
## What is Yorker AI?

Yorker-AI is a research-grade IPL match prediction engine. It ingests ball-by-ball data from Cricsheet, engineers 27 leakage-safe pre-match features, and uses an L1-regularized logistic regression model with recency weighting to predict match winners.

### Current Performance

| Metric | Value |
|---|---|
| 2025 Holdout Accuracy | **67.1%** (70 matches) |
| Walk-Forward Accuracy | **53.5%** (11 seasons) |
| ROC-AUC | **0.704** |
| Brier Score | **0.230** |
| Model Lift over Baseline | **+4.4pp** |

### Features

- 🏏 **27 engineered features**: Elo ratings, venue intelligence, recent form, H2H records, powerplay differentials, season context, streaks
- 🔒 **Leakage-safe**: Pre-match features only. Confirmed XIs separated into a different feature set
- 📊 **Walk-forward validated**: Train on past, predict the future. No data leakage
- 🎯 **Calibrated probabilities**: Brier score and ECE tracking ensure honest probability estimates
- 🏆 **Monte Carlo simulation**: Championship odds based on 2,000 simulated remaining seasons

## Quick Start

```bash
# Run the full pipeline
PYTHONPATH=src python3 -m ipl_ml.cli run-all

# Individual steps
PYTHONPATH=src python3 -m ipl_ml.cli download-data
PYTHONPATH=src python3 -m ipl_ml.cli build-dataset
PYTHONPATH=src python3 -m ipl_ml.cli train
PYTHONPATH=src python3 -m ipl_ml.cli benchmark --target-accuracy 0.97
PYTHONPATH=src python3 -m ipl_ml.cli predict-upcoming
PYTHONPATH=src python3 -m ipl_ml.cli report
```

## Architecture

```
Cricsheet JSON → Feature Engineering → L1 Logistic (recency-weighted) → Calibrated Probabilities
     ↓                    ↓                        ↓                            ↓
 1,100+ matches    27 features         2-season sliding window          Win prob [0, 1]
```

## Tech Stack

- **Python** (pandas, scikit-learn, XGBoost)
- **Data**: Cricsheet (CC-BY-4.0) + Official IPLT20 feeds
- **Dashboard**: Vanilla HTML/CSS/JS (deployed on HuggingFace Spaces)

## Research Caveat

This is a **research project**, not a finished prediction product. The 67.1% holdout accuracy was measured on IPL 2025. Walk-forward accuracy across all seasons is ~53.5%, much closer to baseline. Predictions are probabilistic estimates with wide confidence intervals. Not financial or betting advice.

See [PRD.md](PRD.md) for the full research plan.

## License

MIT
