from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import run_model_benchmark
from .constants import CURRENT_SEASON, EXPLAINER_TOP_N, SIMULATION_RUNS, TEAM_HOME_VENUES
from .data import (
    build_match_table,
    download_cricsheet_archive,
    download_official_schedule,
    extract_zip,
    load_cricsheet_matches,
    schedule_frame,
)
from .features import FeatureBuilder
from .model import explain_fixture, fit_and_persist, score_upcoming
from .reporting import generate_twitter_card, write_summary_json, write_thread


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_download(force: bool = False) -> dict[str, Path]:
    root = project_root()
    raw_dir = root / "data" / "raw"
    archive_path = download_cricsheet_archive(raw_dir, force=force)
    extracted_dir = extract_zip(archive_path, raw_dir / "cricsheet_ipl_json", force=force)
    schedule_path = download_official_schedule(raw_dir, CURRENT_SEASON, force=force)
    return {
        "archive_path": archive_path,
        "extracted_dir": extracted_dir,
        "schedule_path": schedule_path,
    }


def _write_data_quality_reports(dataset: pd.DataFrame, match_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    season_rows = []
    for season, group in match_table.groupby("season"):
        season_rows.append(
            {
                "season": int(season),
                "matches": int(len(group)),
                "completed_matches": int(group["winner"].notna().sum()) if "winner" in group else 0,
                "unique_venues": int(group["venue"].nunique()) if "venue" in group else 0,
                "duplicate_match_ids": int(group["match_id"].duplicated().sum()) if "match_id" in group else 0,
            }
        )
    season_frame = pd.DataFrame(season_rows).sort_values("season")
    season_frame.to_csv(output_dir / "data_quality_by_season.csv", index=False)

    feature_missingness = (
        dataset.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values(["missing_rate", "column"], ascending=[False, True])
    )
    feature_missingness.to_csv(output_dir / "feature_missingness.csv", index=False)

    report = {
        "dataset_rows": int(len(dataset)),
        "dataset_matches": int(dataset["match_id"].nunique()) if "match_id" in dataset else 0,
        "match_table_rows": int(len(match_table)),
        "match_table_matches": int(match_table["match_id"].nunique()) if "match_id" in match_table else 0,
        "duplicate_dataset_rows": int(dataset.duplicated().sum()),
        "duplicate_match_table_ids": int(match_table["match_id"].duplicated().sum()) if "match_id" in match_table else 0,
        "missing_labels": int(dataset["label"].isna().sum()) if "label" in dataset else 0,
        "seasons": season_frame.to_dict(orient="records"),
        "highest_missingness": feature_missingness.head(15).to_dict(orient="records"),
    }
    (output_dir / "data_quality_summary.json").write_text(json.dumps(report, indent=2))


def run_build_dataset(force_download: bool = False) -> dict[str, Any]:
    root = project_root()
    paths = run_download(force=force_download)
    matches = load_cricsheet_matches(paths["extracted_dir"])
    schedule = schedule_frame(paths["schedule_path"])
    feature_builder = FeatureBuilder()
    dataset = feature_builder.build_dataset(matches)
    match_table = build_match_table(matches)

    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed_dir / "match_features.csv", index=False)
    match_table.to_csv(processed_dir / "match_table.csv", index=False)
    schedule.to_csv(processed_dir / "official_schedule_2026.csv", index=False)
    _write_data_quality_reports(dataset, match_table, root / "artifacts" / "data_quality")

    return {
        "matches": matches,
        "schedule": schedule,
        "dataset": dataset,
        "match_table": match_table,
        "feature_builder": feature_builder,
    }


def _points_table_from_match_table(match_table: pd.DataFrame) -> pd.DataFrame:
    season_matches = match_table[match_table["season"] == CURRENT_SEASON].copy()
    if season_matches.empty:
        return pd.DataFrame(columns=["team", "matches", "wins", "losses", "points", "nrr_proxy", "runs_for", "runs_against"])
    season_matches[["team_a_runs", "team_b_runs"]] = season_matches[["team_a_runs", "team_b_runs"]].fillna(0)
    rows = []
    for team in sorted(set(season_matches["team_a"]).union(season_matches["team_b"])):
        team_games = season_matches[(season_matches["team_a"] == team) | (season_matches["team_b"] == team)]
        if team_games.empty:
            continue
        wins = (team_games["winner"] == team).sum()
        losses = ((team_games["winner"].notna()) & (team_games["winner"] != team)).sum()
        no_results = (team_games["result_type"] == "no_result").sum()
        runs_for = np.where(team_games["team_a"] == team, team_games["team_a_runs"], team_games["team_b_runs"]).sum()
        runs_against = np.where(team_games["team_a"] == team, team_games["team_b_runs"], team_games["team_a_runs"]).sum()
        nrr_proxy = (
            np.where(team_games["team_a"] == team, team_games["team_a_runs"] - team_games["team_b_runs"], team_games["team_b_runs"] - team_games["team_a_runs"]).sum()
            / max(len(team_games), 1)
        )
        rows.append(
            {
                "team": team,
                "matches": int(len(team_games)),
                "wins": int(wins),
                "losses": int(losses),
                "points": int(wins * 2 + no_results),
                "nrr_proxy": float(nrr_proxy),
                "runs_for": int(runs_for),
                "runs_against": int(runs_against),
            }
        )
    return pd.DataFrame(rows).sort_values(["points", "nrr_proxy"], ascending=[False, False]).reset_index(drop=True)


def _simulate_season(
    current_table: pd.DataFrame,
    upcoming_predictions: pd.DataFrame,
    feature_builder: FeatureBuilder,
    live_model: Any,
) -> pd.DataFrame:
    if current_table.empty or upcoming_predictions.empty:
        return pd.DataFrame(columns=["team", "top4_probability", "champion_probability"])

    teams = current_table["team"].tolist()
    top4_counts = {team: 0 for team in teams}
    champion_counts = {team: 0 for team in teams}
    base_points = current_table.set_index("team")["points"].to_dict()
    base_nrr = current_table.set_index("team")["nrr_proxy"].to_dict()
    rng = np.random.default_rng(42)

    def playoff_venue(team: str, fallback: str = "Narendra Modi Stadium") -> str:
        venues = sorted(TEAM_HOME_VENUES.get(team, {fallback}))
        return venues[0] if venues else fallback

    cache: dict[tuple[str, str, str], float] = {}

    def matchup_prob(team_a: str, team_b: str, venue: str, when: pd.Timestamp) -> float:
        key = (team_a, team_b, venue)
        if key in cache:
            return cache[key]
        fixture = pd.DataFrame(
            [
                {
                    "match_id": f"sim-{team_a}-{team_b}-{venue}",
                    "date": when,
                    "season": CURRENT_SEASON,
                    "team_a": team_a,
                    "team_b": team_b,
                    "venue": venue,
                }
            ]
        )
        temp = feature_builder.upcoming_rows(fixture)
        if temp.empty:
            cache[key] = 0.5
            return cache[key]
        cache[key] = float(live_model.predict_proba(temp.drop(columns=["label"], errors="ignore"))[:, 1][0])
        return cache[key]

    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            matchup_prob(team_a, team_b, playoff_venue(team_a), pd.Timestamp("2026-05-26"))
            matchup_prob(team_a, team_b, "Narendra Modi Stadium", pd.Timestamp("2026-05-31"))

    for _ in range(SIMULATION_RUNS):
        points = base_points.copy()
        nrr = base_nrr.copy()
        for row in upcoming_predictions.itertuples(index=False):
            p_team_a = float(row.predicted_prob_team_a)
            winner = row.team_a if rng.random() < p_team_a else row.team_b
            loser = row.team_b if winner == row.team_a else row.team_a
            points[winner] = points.get(winner, 0) + 2
            nrr[winner] = nrr.get(winner, 0.0) + abs(p_team_a - 0.5) * 0.1
            nrr[loser] = nrr.get(loser, 0.0) - abs(p_team_a - 0.5) * 0.1

        ranked = sorted(teams, key=lambda team: (points[team], nrr.get(team, 0.0)), reverse=True)
        top4 = ranked[:4]
        for team in top4:
            top4_counts[team] += 1

        q1_prob = matchup_prob(top4[0], top4[1], playoff_venue(top4[0]), pd.Timestamp("2026-05-26"))
        elim_prob = matchup_prob(top4[2], top4[3], playoff_venue(top4[2]), pd.Timestamp("2026-05-27"))
        q1_winner = top4[0] if rng.random() < q1_prob else top4[1]
        q1_loser = top4[1] if q1_winner == top4[0] else top4[0]
        elim_winner = top4[2] if rng.random() < elim_prob else top4[3]
        q2_prob = matchup_prob(q1_loser, elim_winner, playoff_venue(q1_loser), pd.Timestamp("2026-05-29"))
        q2_winner = q1_loser if rng.random() < q2_prob else elim_winner
        final_prob = matchup_prob(q1_winner, q2_winner, "Narendra Modi Stadium", pd.Timestamp("2026-05-31"))
        champion = q1_winner if rng.random() < final_prob else q2_winner
        champion_counts[champion] += 1

    rows = []
    for team in teams:
        rows.append(
            {
                "team": team,
                "top4_probability": top4_counts[team] / SIMULATION_RUNS,
                "champion_probability": champion_counts[team] / SIMULATION_RUNS,
            }
        )
    return pd.DataFrame(rows).sort_values(["champion_probability", "top4_probability"], ascending=[False, False])


def run_train(force_download: bool = False) -> dict[str, Any]:
    root = project_root()
    built = run_build_dataset(force_download=force_download)
    model_dir = root / "artifacts" / "models"
    fitted = fit_and_persist(built["dataset"], model_dir)
    return {**built, **fitted}


def run_benchmark(force_download: bool = False, target_accuracy: float = 0.97) -> dict[str, Any]:
    root = project_root()
    built = run_build_dataset(force_download=force_download)
    benchmark = run_model_benchmark(
        built["dataset"],
        root / "artifacts" / "models",
        target_accuracy=target_accuracy,
    )
    return {**built, "benchmark": benchmark}


def run_predict_upcoming(force_download: bool = False) -> dict[str, Any]:
    trained = run_train(force_download=force_download)
    schedule = trained["schedule"]
    today = pd.Timestamp.today().normalize()
    upcoming_fixtures = schedule[
        (schedule["season"] == CURRENT_SEASON)
        & (schedule["match_status"].str.lower() == "upcoming")
        & (pd.to_datetime(schedule["date"]) >= today)
    ].copy()
    upcoming_feature_rows = trained["feature_builder"].upcoming_rows(upcoming_fixtures)
    scored = score_upcoming(trained["live_model"], upcoming_feature_rows)
    scored.to_csv(project_root() / "artifacts" / "models" / "upcoming_predictions.csv", index=False)

    explanations = {}
    for row in upcoming_feature_rows.head(5).itertuples(index=False):
        feature_row = upcoming_feature_rows[
            (upcoming_feature_rows["match_id"] == row.match_id)
            & (upcoming_feature_rows["team_a"] == row.team_a)
            & (upcoming_feature_rows["team_b"] == row.team_b)
        ].drop(columns=["label"])
        explanations[f"{row.team_a} vs {row.team_b}"] = explain_fixture(
            trained["explainer_model"],
            feature_row,
            top_n=EXPLAINER_TOP_N,
        )
    write_summary_json(explanations, project_root() / "artifacts" / "models" / "fixture_explanations.json")

    current_table = _points_table_from_match_table(trained["match_table"])
    current_table.to_csv(project_root() / "artifacts" / "models" / "current_points_table.csv", index=False)
    season_sim = _simulate_season(current_table, scored, trained["feature_builder"], trained["live_model"])
    season_sim.to_csv(project_root() / "artifacts" / "models" / "season_simulation.csv", index=False)
    write_summary_json(
        season_sim.to_dict(orient="records"),
        project_root() / "artifacts" / "models" / "season_simulation.json",
    )

    return {
        **trained,
        "upcoming_feature_rows": upcoming_feature_rows,
        "upcoming_predictions": scored,
        "current_table": current_table,
        "season_simulation": season_sim,
    }


def run_report(force_download: bool = False) -> dict[str, Any]:
    predicted = run_predict_upcoming(force_download=force_download)
    twitter_dir = project_root() / "reports" / "twitter"
    generate_twitter_card(
        predicted["metrics"],
        predicted["upcoming_predictions"],
        predicted["season_simulation"],
        twitter_dir / "twitter_card.png",
    )
    write_thread(
        predicted["metrics"],
        predicted["upcoming_predictions"],
        predicted["season_simulation"],
        twitter_dir / "thread.md",
    )
    summary = {
        "metrics": predicted["metrics"],
        "upcoming_head": predicted["upcoming_predictions"].head(10).to_dict(orient="records"),
        "season_simulation_head": predicted["season_simulation"].head(10).to_dict(orient="records"),
    }
    write_summary_json(summary, twitter_dir / "summary.json")
    return predicted


def run_all(force_download: bool = False) -> dict[str, Any]:
    return run_report(force_download=force_download)
