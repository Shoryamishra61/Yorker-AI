from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .constants import VALIDATION_SEASON
from .model import (
    QUALITY_MODEL_NAME,
    build_preprocessor,
    build_quality_model,
    classification_metrics,
    quality_feature_columns,
    selected_feature_columns,
)


def _model_pipeline(name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    if name == QUALITY_MODEL_NAME:
        return build_quality_model(numeric_features, categorical_features)
    if name == "regularized_logistic":
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", LogisticRegression(max_iter=2500, C=0.05, random_state=42)),
            ]
        )
    if name == "regularized_logistic_with_identity":
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", LogisticRegression(max_iter=2500, C=0.03, random_state=42)),
            ]
        )
    if name == "shallow_random_forest":
        return Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=2,
                        min_samples_leaf=8,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown benchmark model: {name}")


def _fit_with_recency_weights(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, seasons: pd.Series) -> None:
    step_name = list(model.named_steps)[-1]
    weights = 1.0 + 0.2 * (seasons - seasons.min())
    try:
        model.fit(x_train, y_train, **{f"{step_name}__sample_weight": weights})
    except TypeError:
        model.fit(x_train, y_train)


def _baseline_probabilities(frame: pd.DataFrame, feature: str) -> np.ndarray:
    values = frame[feature].to_numpy(dtype=float)
    if feature.endswith("_expected_team_a"):
        return np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.where(values >= 0.0, 0.55, 0.45)


def _row(
    *,
    candidate: str,
    season: int,
    kind: str,
    train_start_season: int | None,
    train_end_season: int | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "season": int(season),
        "kind": kind,
        "train_start_season": train_start_season,
        "train_end_season": train_end_season,
        **metrics,
    }


def _write_benchmark_report(
    output_dir: Path,
    by_season: pd.DataFrame,
    summary_frame: pd.DataFrame,
    target_gate: dict[str, Any],
) -> None:
    lines = [
        "# Benchmark Report",
        "",
        "This report is generated from leakage-aware walk-forward validation.",
        "",
        "## Quality Gate",
        "",
        f"- Candidate: `{target_gate['candidate']}`",
        f"- Validation season: {target_gate['validation_season']}",
        f"- Target accuracy: {target_gate['target_accuracy']:.3f}",
        f"- Observed accuracy: {target_gate['observed_accuracy']:.3f}" if target_gate["observed_accuracy"] is not None else "- Observed accuracy: n/a",
        f"- 95% CI: [{target_gate['accuracy_ci95_low']:.3f}, {target_gate['accuracy_ci95_high']:.3f}]" if target_gate["accuracy_ci95_low"] is not None else "- 95% CI: n/a",
        f"- Passed: {str(target_gate['passed']).lower()}",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Feature Set | Kind | Seasons | Weighted Accuracy | Fixture Accuracy | ROC-AUC | Brier | Log Loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_frame.to_dict(orient="records"):
        feature_set = row.get("feature_set") or ""
        lines.append(
            f"| {row['candidate']} | {feature_set} | {row['kind']} | {int(row['seasons'])} | "
            f"{row['weighted_accuracy']:.3f} | {row['weighted_fixture_accuracy']:.3f} | "
            f"{row['mean_roc_auc']:.3f} | {row['mean_brier_score']:.3f} | {row['mean_log_loss']:.3f} |"
        )

    model_rows = summary_frame[summary_frame["kind"] == "model"].copy()
    baseline_rows = summary_frame[summary_frame["kind"] == "baseline"].copy()
    lines.extend(["", "## Credibility Notes", ""])
    if not model_rows.empty and not baseline_rows.empty:
        best_model = model_rows.iloc[0]
        best_baseline = baseline_rows.iloc[0]
        lift = float(best_model["weighted_accuracy"] - best_baseline["weighted_accuracy"])
        lines.append(f"- Best model by weighted accuracy: `{best_model['candidate']}` at {best_model['weighted_accuracy']:.3f}.")
        lines.append(f"- Best baseline by weighted accuracy: `{best_baseline['candidate']}` at {best_baseline['weighted_accuracy']:.3f}.")
        lines.append(f"- Model lift over best baseline: {lift:+.3f}.")
    lines.append("- `prematch_core` excludes confirmed-XI player aggregate features.")
    lines.append("- `toss_confirmed_xi` includes features that are only defensible after teams are known.")
    lines.append("- Public claims should use walk-forward metrics, not only a single validation season.")

    lines.extend(["", "## Per-Season Primary Candidate", ""])
    primary = by_season[by_season["candidate"] == target_gate["candidate"]].sort_values("season")
    lines.append("| Season | Accuracy | Fixture Accuracy | ROC-AUC | Brier | Validation Matches |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in primary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['season'])} | {row['accuracy']:.3f} | {row['fixture_accuracy']:.3f} | {row['roc_auc']:.3f} | {row['brier_score']:.3f} | {int(row['validation_matches'])} |"
        )

    (output_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n")


def run_model_benchmark(
    dataset: pd.DataFrame,
    output_dir: Path,
    target_accuracy: float = 0.97,
    start_validation_season: int = 2016,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seasons = [
        int(season)
        for season in sorted(dataset["season"].unique())
        if int(season) >= start_validation_season
    ]

    feature_sets = ["prematch_core", "toss_confirmed_xi"]
    candidate_specs = [
        {"candidate": QUALITY_MODEL_NAME, "kind": "model", "window": 2, "include_categorical": False},
        {"candidate": "regularized_logistic", "kind": "model", "window": 5, "include_categorical": False},
        {"candidate": "regularized_logistic_with_identity", "kind": "model", "window": 5, "include_categorical": True},
        {"candidate": "shallow_random_forest", "kind": "model", "window": 2, "include_categorical": False},
    ]
    baseline_features = [
        "recent_win_rate_diff",
        "season_win_rate_diff",
        "venue_win_rate_diff",
        "elo_expected_team_a",
        "elo_home_expected_team_a",
    ]

    rows: list[dict[str, Any]] = []
    for season in validation_seasons:
        valid_frame = dataset[dataset["season"] == season].copy()
        if valid_frame.empty:
            continue

        for feature_set in feature_sets:
            for spec in candidate_specs:
                numeric_features, categorical_features = selected_feature_columns(
                    dataset,
                    feature_set=feature_set,
                    include_categorical=bool(spec["include_categorical"]),
                )
                window = int(spec["window"])
                train_start = season - window
                train_end = season - 1
                train_frame = dataset[
                    (dataset["season"] >= train_start) & (dataset["season"] <= train_end)
                ].copy()
                if train_frame.empty:
                    continue
                model = _model_pipeline(spec["candidate"], numeric_features, categorical_features)
                x_train = train_frame.drop(columns=["label"])
                y_train = train_frame["label"].astype(int)
                _fit_with_recency_weights(model, x_train, y_train, train_frame["season"])
                probabilities = model.predict_proba(valid_frame.drop(columns=["label"]))[:, 1]
                metrics = classification_metrics(valid_frame, probabilities)
                rows.append(
                    _row(
                        candidate=f"{spec['candidate']}__{feature_set}",
                        season=season,
                        kind="model",
                        train_start_season=int(train_start),
                        train_end_season=int(train_end),
                        metrics={**metrics, "feature_set": feature_set},
                    )
                )

        for feature in baseline_features:
            if feature not in valid_frame.columns:
                continue
            probabilities = _baseline_probabilities(valid_frame, feature)
            metrics = classification_metrics(valid_frame, probabilities)
            rows.append(
                _row(
                    candidate=f"baseline_{feature}",
                    season=season,
                    kind="baseline",
                    train_start_season=None,
                    train_end_season=None,
                    metrics={**metrics, "feature_set": "single_feature_baseline"},
                )
            )

    by_season = pd.DataFrame(rows)
    by_season.to_csv(output_dir / "benchmark_by_season.csv", index=False)

    summaries = []
    for candidate, group in by_season.groupby("candidate"):
        validation_rows = group["validation_rows"].astype(float)
        validation_matches = group["validation_matches"].astype(float)
        summaries.append(
            {
                "candidate": candidate,
                "kind": group["kind"].iloc[0],
                "feature_set": group["feature_set"].iloc[0] if "feature_set" in group else None,
                "seasons": int(group["season"].nunique()),
                "weighted_accuracy": float(np.average(group["accuracy"], weights=validation_rows)),
                "mean_season_accuracy": float(group["accuracy"].mean()),
                "weighted_fixture_accuracy": float(np.average(group["fixture_accuracy"], weights=validation_matches)),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "mean_log_loss": float(group["log_loss"].mean()),
                "mean_brier_score": float(group["brier_score"].mean()),
            }
        )

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["weighted_accuracy", "mean_roc_auc"],
        ascending=[False, False],
    )
    primary_rows = by_season[
        (by_season["candidate"] == f"{QUALITY_MODEL_NAME}__prematch_core")
        & (by_season["season"] == VALIDATION_SEASON)
    ]
    primary = primary_rows.iloc[0].to_dict() if not primary_rows.empty else {}
    target_gate = {
        "target_accuracy": float(target_accuracy),
        "candidate": f"{QUALITY_MODEL_NAME}__prematch_core",
        "validation_season": int(VALIDATION_SEASON),
        "observed_accuracy": float(primary["accuracy"]) if primary else None,
        "accuracy_ci95_low": float(primary["accuracy_ci95_low"]) if primary else None,
        "accuracy_ci95_high": float(primary["accuracy_ci95_high"]) if primary else None,
        "passed": bool(primary and float(primary["accuracy"]) >= target_accuracy),
    }

    result = {
        "target_gate": target_gate,
        "candidate_summary": summary_frame.to_dict(orient="records"),
        "validation_season_rows": by_season.to_dict(orient="records"),
    }
    (output_dir / "benchmark_metrics.json").write_text(json.dumps(result, indent=2))
    _write_benchmark_report(output_dir, by_season, summary_frame, target_gate)
    return result
