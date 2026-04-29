I am building an IPL match prediction research project.

Current strict pre-match validation:
- 2025 holdout accuracy: 67.1%
Brier score: 0.230
ROC-AUC: 0.704
ECE: 0.114

Current model uses only pre-match-safe numeric features:
- team form before the match
- venue priors and home advantage
- team momentum and NRR trend
- head-to-head history
- Elo-style strength

Confirmed-XI player features are now separated into a toss-confirmed feature set, because using known XIs for historical rows is not a fair match-eve setup.

Next research call: Royal Challengers Bengaluru vs Gujarat Titans
Model lean: Gujarat Titans at 50.4%
Venue: M. Chinnaswamy Stadium | Date: 24 Apr 2026

Research artifacts generated:
- validation predictions
- feature availability audit
- walk-forward benchmark report
- data quality summary
- upcoming fixture probabilities with caveats
