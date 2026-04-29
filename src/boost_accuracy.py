"""
Accuracy Booster — Systematic search for best model configuration.
Adds interaction features, polynomial terms, and tests multiple architectures.
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, VotingClassifier,
    ExtraTreesClassifier, BaggingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ipl_ml.model import feature_columns, REFERENCE_COLUMNS, fixture_accuracy

# Load dataset
dataset = pd.read_csv(ROOT / "data" / "processed" / "match_features.csv")
dataset["date"] = pd.to_datetime(dataset["date"])

numeric_cols = [c for c in dataset.columns if c not in REFERENCE_COLUMNS
                and pd.api.types.is_numeric_dtype(dataset[c])]

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add high-value interaction features."""
    df = df.copy()
    # Key interactions that capture real cricket dynamics
    df["elo_x_home"] = df["elo_diff"] * df["home_advantage_diff"]
    df["form_x_venue"] = df["recent_win_rate_diff"] * df["venue_win_rate_diff"]
    df["batting_x_bowling"] = df["batting_power_diff"] * df["bowling_danger_diff"]
    df["streak_x_form"] = df["current_streak_diff"] * df["recent_win_rate_diff"]
    df["elo_x_h2h"] = df["elo_diff"] * df["head_to_head_diff"]
    df["xi_bat_x_bowl"] = df["xi_batting_form_diff"] * df["xi_bowling_form_diff"]
    df["nrr_x_form"] = df["season_nrr_diff"] * df["recent_win_rate_diff"]
    df["pp_runs_x_wkts"] = df["powerplay_runs_diff"] * df["powerplay_wickets_diff"]
    df["venue_score_x_chase"] = df["venue_avg_first_innings_score"] * df["venue_chase_win_pct"]
    df["experience_x_form"] = df["xi_experience_diff"] * df["xi_batting_form_diff"]
    df["matchup_edge"] = df["xi_batting_vs_opponent_diff"] - df["xi_bowling_vs_opponent_diff"]
    df["combined_strength"] = (df["elo_diff"] / 100 + df["recent_win_rate_diff"] +
                                df["batting_power_diff"] + df["bowling_danger_diff"]) / 4
    return df

dataset = add_interaction_features(dataset)
numeric_cols_ext = [c for c in dataset.columns if c not in REFERENCE_COLUMNS
                    and pd.api.types.is_numeric_dtype(dataset[c])]

print(f"Features: {len(numeric_cols)} base → {len(numeric_cols_ext)} with interactions")
print()

# Test configurations
configs = []

for window in [2, 3, 4, 5, 6]:
    val = dataset[dataset["season"] == 2025].copy()
    train_start = 2025 - window
    train = dataset[(dataset["season"] >= train_start) & (dataset["season"] <= 2024)].copy()
    
    if train.empty or val.empty:
        continue
    
    X_train = train[numeric_cols_ext].fillna(0)
    y_train = train["label"].astype(int)
    X_val = val[numeric_cols_ext].fillna(0)
    y_val = val["label"].astype(int)
    weights = (1.0 + 0.2 * (train["season"] - train["season"].min())).values
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    models = {
        f"LR_C0.1_w{window}": LogisticRegression(C=0.1, penalty="l1", solver="liblinear", max_iter=3000, random_state=42),
        f"LR_C0.05_w{window}": LogisticRegression(C=0.05, penalty="l1", solver="liblinear", max_iter=3000, random_state=42),
        f"LR_C0.2_w{window}": LogisticRegression(C=0.2, penalty="l1", solver="liblinear", max_iter=3000, random_state=42),
        f"LR_C0.5_w{window}": LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=3000, random_state=42),
        f"RF_d4_w{window}": RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=8, random_state=42),
        f"RF_d3_w{window}": RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=12, random_state=42),
        f"HGB_w{window}": HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.03, min_samples_leaf=15, l2_regularization=2.0, random_state=42),
        f"GBM_w{window}": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8, min_samples_leaf=10, random_state=42),
        f"ET_w{window}": ExtraTreesClassifier(n_estimators=400, max_depth=3, min_samples_leaf=10, random_state=42),
    }
    
    if HAS_XGB:
        models[f"XGB_w{window}"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
            reg_alpha=0.5, min_child_weight=5, random_state=42,
            eval_metric="logloss",
        )
    
    for name, model in models.items():
        try:
            if "sample_weight" in model.fit.__code__.co_varnames:
                model.fit(X_train_s, y_train, sample_weight=weights)
            else:
                model.fit(X_train_s, y_train)
        except TypeError:
            model.fit(X_train_s, y_train)
        
        probs = model.predict_proba(X_val_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        
        # Fixture accuracy
        val_check = val[["match_id", "team_a", "team_b", "label"]].copy()
        val_check["probability"] = probs
        fix_acc = fixture_accuracy(val_check, probs)
        
        configs.append({
            "name": name, "window": window,
            "train_rows": len(train), "accuracy": acc,
            "fixture_accuracy": fix_acc, "roc_auc": auc,
        })

# Sort by accuracy
configs.sort(key=lambda x: x["accuracy"], reverse=True)

print("=" * 80)
print(f"{'Model':<30} {'Window':>6} {'Train':>6} {'Acc':>8} {'Fix Acc':>8} {'AUC':>8}")
print("=" * 80)
for c in configs[:25]:
    print(f"{c['name']:<30} {c['window']:>6} {c['train_rows']:>6} {c['accuracy']:>7.1%} {c['fixture_accuracy']:>7.1%} {c['roc_auc']:>7.3f}")

print()
print(f"BEST: {configs[0]['name']} → {configs[0]['accuracy']:.1%} accuracy, {configs[0]['fixture_accuracy']:.1%} fixture")
print()

# Save best config
best = configs[0]
with open(ROOT / "artifacts" / "models" / "best_config.json", "w") as f:
    json.dump(best, f, indent=2)
