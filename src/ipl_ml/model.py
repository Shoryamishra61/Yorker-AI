from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import inspect

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_registry import REFERENCE_COLUMNS, columns_for_feature_set, leakage_report

try:
    from xgboost import DMatrix, XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - environment dependent
    DMatrix = None
    XGBClassifier = None
    HAS_XGBOOST = False

from .constants import CURRENT_SEASON, TRAIN_END_SEASON, VALIDATION_SEASON


QUALITY_TRAIN_WINDOW_SEASONS = 2
QUALITY_MODEL_NAME = "recency_l1_logistic"


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = []
    categorical = []
    for column in frame.columns:
        if column in REFERENCE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    encoder_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(**encoder_kwargs)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_stack_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    base_estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_depth=5,
                max_iter=300,
                learning_rate=0.04,
                min_samples_leaf=8,
                l2_regularization=0.5,
                random_state=42,
            ),
        ),
        (
            "lr",
            LogisticRegression(max_iter=2000, C=0.3, random_state=42),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
    if HAS_XGBOOST:
        base_estimators.insert(
            0,
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    reg_alpha=0.3,
                    min_child_weight=3,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        )
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        stack_method="predict_proba",
        passthrough=True,
        cv=5,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", stacker)])


def selected_feature_columns(
    frame: pd.DataFrame,
    feature_set: str = "prematch_core",
    include_categorical: bool = False,
) -> tuple[list[str], list[str]]:
    selected_columns = columns_for_feature_set(list(frame.columns), feature_set)
    selected = frame[selected_columns]
    numeric_features, categorical_features = feature_columns(selected)
    return numeric_features, categorical_features if include_categorical else []


def quality_feature_columns(frame: pd.DataFrame, feature_set: str = "prematch_core") -> tuple[list[str], list[str]]:
    return selected_feature_columns(frame, feature_set=feature_set, include_categorical=False)


def write_feature_audit(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature_set in ["prematch_core", "toss_confirmed_xi"]:
        numeric_features, categorical_features = selected_feature_columns(
            frame,
            feature_set=feature_set,
            include_categorical=False,
        )
        selected = set(numeric_features + categorical_features)
        rows.extend(
            {"feature_set": feature_set, "used_by_selected_model": str(row["column"] in selected).lower(), **row}
            for row in leakage_report(list(frame.columns), feature_set)
        )
    pd.DataFrame(rows).to_csv(output_dir / "feature_availability_audit.csv", index=False)


def build_quality_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """L1 logistic with 2-season recency window."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = LogisticRegression(
        max_iter=2500,
        C=0.1,
        penalty="l1",
        solver="liblinear",
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_explainer_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    if HAS_XGBOOST:
        explainer = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    else:
        explainer = LogisticRegression(max_iter=1500, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", explainer)])


def recency_sample_weights(seasons: pd.Series) -> np.ndarray:
    return (1.0 + 0.2 * (seasons - seasons.min())).to_numpy(dtype=float)


def fit_with_recency_weights(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, seasons: pd.Series) -> None:
    step_name = list(model.named_steps)[-1]
    weights = recency_sample_weights(seasons)
    try:
        model.fit(x_train, y_train, **{f"{step_name}__sample_weight": weights})
    except TypeError:
        model.fit(x_train, y_train)


def expected_calibration_error(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> float:
    y_values = y_true.to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean() * abs(probabilities[mask].mean() - y_values[mask].mean()))
    return ece


def fixture_accuracy(validation_frame: pd.DataFrame, probabilities: np.ndarray) -> float | None:
    frame = validation_frame[["match_id", "team_a", "team_b", "label"]].copy()
    frame["probability"] = probabilities
    correct = 0
    total = 0
    for _, group in frame.groupby("match_id"):
        if len(group) != 2:
            continue
        first = group.iloc[0]
        second = group.iloc[1]
        predicted_winner = first.team_a if first.probability >= second.probability else second.team_a
        actual_winner = first.team_a if int(first.label) == 1 else first.team_b
        correct += int(predicted_winner == actual_winner)
        total += 1
    if total == 0:
        return None
    return float(correct / total)


def classification_metrics(validation_frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float | int | None]:
    y_valid = validation_frame["label"].astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_valid, predictions))
    n = int(len(validation_frame))
    standard_error = float(np.sqrt(accuracy * (1.0 - accuracy) / max(n, 1)))
    return {
        "accuracy": accuracy,
        "accuracy_ci95_low": float(max(0.0, accuracy - 1.96 * standard_error)),
        "accuracy_ci95_high": float(min(1.0, accuracy + 1.96 * standard_error)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_valid, probabilities)),
        "log_loss": float(log_loss(y_valid, np.clip(probabilities, 1e-6, 1 - 1e-6))),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "expected_calibration_error": expected_calibration_error(y_valid, probabilities),
        "mean_confidence": float(np.maximum(probabilities, 1.0 - probabilities).mean()),
        "fixture_accuracy": fixture_accuracy(validation_frame, probabilities),
        "validation_rows": n,
        "validation_matches": int(validation_frame["match_id"].nunique()),
    }


def evaluate_model(model: Pipeline, validation_frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    x_valid = validation_frame.drop(columns=["label"])
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(validation_frame, probabilities)
    prediction_frame = validation_frame[["match_id", "date", "season", "team_a", "team_b", "venue", "label"]].copy()
    prediction_frame["predicted_prob_team_a"] = probabilities
    prediction_frame["predicted_label"] = predictions
    return metrics, prediction_frame


def fit_and_persist(dataset: pd.DataFrame, model_dir: Path) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_set = "prematch_core"
    numeric_features, categorical_features = quality_feature_columns(dataset, feature_set=feature_set)
    write_feature_audit(dataset, model_dir)

    train_start_season = max(int(dataset["season"].min()), TRAIN_END_SEASON - QUALITY_TRAIN_WINDOW_SEASONS + 1)
    train_frame = dataset[
        (dataset["season"] >= train_start_season) & (dataset["season"] <= TRAIN_END_SEASON)
    ].copy()
    valid_frame = dataset[dataset["season"] == VALIDATION_SEASON].copy()
    x_train = train_frame.drop(columns=["label"])
    y_train = train_frame["label"].astype(int)

    validation_model = build_quality_model(numeric_features, categorical_features)
    fit_with_recency_weights(validation_model, x_train, y_train, train_frame["season"])
    metrics, validation_predictions = evaluate_model(validation_model, valid_frame)
    metrics.update(
        {
            "model_name": QUALITY_MODEL_NAME,
            "train_start_season": int(train_start_season),
            "train_end_season": int(TRAIN_END_SEASON),
            "train_rows": int(len(train_frame)),
        }
    )

    live_end_season = min(int(dataset["season"].max()), CURRENT_SEASON)
    live_start_season = max(int(dataset["season"].min()), live_end_season - QUALITY_TRAIN_WINDOW_SEASONS + 1)
    live_frame = dataset[
        (dataset["season"] >= live_start_season) & (dataset["season"] <= live_end_season)
    ].copy()
    x_live = live_frame.drop(columns=["label"])
    y_live = live_frame["label"].astype(int)
    live_model = build_quality_model(numeric_features, categorical_features)
    fit_with_recency_weights(live_model, x_live, y_live, live_frame["season"])

    explainer_model = build_explainer_model(numeric_features, categorical_features)
    fit_with_recency_weights(explainer_model, x_live, y_live, live_frame["season"])

    joblib.dump(live_model, model_dir / "stack_model.joblib")
    joblib.dump(explainer_model, model_dir / "explainer_model.joblib")
    (model_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "model_name": QUALITY_MODEL_NAME,
                "feature_set": feature_set,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "validation_train_start_season": train_start_season,
                "validation_train_end_season": TRAIN_END_SEASON,
                "live_train_start_season": live_start_season,
                "live_train_end_season": live_end_season,
            },
            indent=2,
        )
    )
    (model_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    validation_predictions.to_csv(model_dir / "validation_predictions.csv", index=False)

    return {
        "live_model": live_model,
        "explainer_model": explainer_model,
        "metrics": metrics,
        "validation_predictions": validation_predictions,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def score_upcoming(model: Pipeline, fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures.copy()
    features = fixtures.drop(columns=["label"], errors="ignore")
    probabilities = model.predict_proba(features)[:, 1]
    scored = fixtures[["match_id", "date", "season", "team_a", "team_b", "venue"]].copy()
    scored["predicted_prob_team_a"] = probabilities
    scored["predicted_prob_team_b"] = 1.0 - probabilities
    scored["predicted_winner"] = np.where(
        scored["predicted_prob_team_a"] >= 0.5,
        scored["team_a"],
        scored["team_b"],
    )
    scored["confidence"] = np.maximum(scored["predicted_prob_team_a"], scored["predicted_prob_team_b"])
    return scored.sort_values(["date", "match_id"])


def explain_fixture(explainer_model: Pipeline, fixture_row: pd.DataFrame, top_n: int = 5) -> list[dict[str, float]]:
    preprocessor = explainer_model.named_steps["preprocessor"]
    explainer = explainer_model.named_steps["model"]
    transformed = preprocessor.transform(fixture_row)
    feature_names = preprocessor.get_feature_names_out()
    if HAS_XGBOOST and hasattr(explainer, "get_booster"):
        dmatrix = DMatrix(transformed, feature_names=feature_names.tolist())
        contributions = explainer.get_booster().predict(dmatrix, pred_contribs=True)[0][:-1]
    elif hasattr(explainer, "coef_"):
        contributions = transformed[0] * explainer.coef_[0]
    else:
        importances = getattr(explainer, "feature_importances_", np.zeros(len(feature_names)))
        contributions = transformed[0] * importances
    pairs = [
        {"feature": feature_names[idx], "contribution": float(value)}
        for idx, value in enumerate(contributions)
    ]
    pairs.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return pairs[:top_n]
