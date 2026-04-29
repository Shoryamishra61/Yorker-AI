from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Availability = Literal["preseason", "match_eve", "toss_confirmed", "post_match", "reference"]


@dataclass(frozen=True)
class FeatureInfo:
    availability: Availability
    family: str
    description: str


REFERENCE_COLUMNS = {"match_id", "date", "label"}

FEATURE_REGISTRY: dict[str, FeatureInfo] = {
    "match_id": FeatureInfo("reference", "identity", "Stable match identifier."),
    "date": FeatureInfo("reference", "identity", "Scheduled match date."),
    "label": FeatureInfo("post_match", "target", "Whether team_a won the match."),
    "season": FeatureInfo("preseason", "schedule", "Tournament season."),
    "venue": FeatureInfo("match_eve", "schedule", "Scheduled venue."),
    "team_a": FeatureInfo("match_eve", "schedule", "Perspective team for the row."),
    "team_b": FeatureInfo("match_eve", "schedule", "Opponent team for the row."),
    "is_playoff": FeatureInfo("match_eve", "schedule", "Whether fixture is a playoff match."),
    "month": FeatureInfo("match_eve", "schedule", "Calendar month of fixture."),
    "elo_diff": FeatureInfo("match_eve", "team_strength", "Pre-match Elo difference."),
    "elo_expected_team_a": FeatureInfo("match_eve", "team_strength", "Pre-match Elo win probability."),
    "elo_home_expected_team_a": FeatureInfo("match_eve", "team_strength", "Home-adjusted pre-match Elo probability."),
    "season_win_rate_diff": FeatureInfo("match_eve", "team_form", "Current-season win-rate difference before match."),
    "season_nrr_diff": FeatureInfo("match_eve", "team_form", "Current-season run-rate proxy difference before match."),
    "season_runs_for_diff": FeatureInfo("match_eve", "team_form", "Current-season runs-for average difference before match."),
    "season_runs_against_diff": FeatureInfo("match_eve", "team_form", "Current-season runs-against average difference before match."),
    "season_matches_played_diff": FeatureInfo("match_eve", "team_form", "Current-season matches-played difference before match."),
    "current_streak_diff": FeatureInfo("match_eve", "team_form", "Signed current streak difference before match."),
    "days_since_last_match_diff": FeatureInfo("match_eve", "rest", "Rest-day difference before match."),
    "recent_win_rate_diff": FeatureInfo("match_eve", "team_form", "Last-five win-rate difference before match."),
    "historical_win_rate_diff": FeatureInfo("match_eve", "team_strength", "Historical win-rate difference before match."),
    "recent_nrr_diff": FeatureInfo("match_eve", "team_form", "Last-five run-rate proxy difference before match."),
    "recent_runs_for_diff": FeatureInfo("match_eve", "team_form", "Last-five runs-for difference before match."),
    "recent_runs_against_diff": FeatureInfo("match_eve", "team_form", "Last-five runs-against difference before match."),
    "powerplay_runs_diff": FeatureInfo("match_eve", "phase", "Recent powerplay scoring difference before match."),
    "powerplay_wickets_diff": FeatureInfo("match_eve", "phase", "Recent powerplay wickets-lost difference before match."),
    "venue_win_rate_diff": FeatureInfo("match_eve", "venue", "Team venue win-rate difference before match."),
    "head_to_head_diff": FeatureInfo("match_eve", "head_to_head", "Historical head-to-head win-rate difference before match."),
    "home_advantage_diff": FeatureInfo("match_eve", "venue", "Home-venue indicator difference."),
    "venue_avg_first_innings_score": FeatureInfo("preseason", "venue", "Static venue first-innings scoring prior."),
    "venue_chase_win_pct": FeatureInfo("preseason", "venue", "Static venue chase-win prior."),
    "venue_spin_bias": FeatureInfo("preseason", "venue", "Static venue spin-bowling prior."),
    "venue_pace_bias": FeatureInfo("preseason", "venue", "Static venue pace-bowling prior."),
    "batting_power_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI batting form difference."),
    "bowling_danger_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI bowling form difference."),
    "xi_experience_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI experience difference."),
    "xi_batting_form_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI batting form difference."),
    "xi_bowling_form_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI bowling form difference."),
    "xi_batting_vs_opponent_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI batter-vs-opponent proxy."),
    "xi_bowling_vs_opponent_diff": FeatureInfo("toss_confirmed", "player_xi", "Confirmed-XI bowler-vs-opponent proxy."),
    "known_xi_ratio_diff": FeatureInfo("toss_confirmed", "player_xi", "Known-player ratio difference in confirmed XI."),
}

FEATURE_SET_MAX_AVAILABILITY: dict[str, set[Availability]] = {
    "prematch_core": {"preseason", "match_eve", "reference"},
    "toss_confirmed_xi": {"preseason", "match_eve", "toss_confirmed", "reference"},
}


def feature_info(column: str) -> FeatureInfo:
    return FEATURE_REGISTRY.get(
        column,
        FeatureInfo("toss_confirmed", "uncatalogued", "Uncatalogued feature; excluded from strict pre-match sets."),
    )


def columns_for_feature_set(columns: list[str], feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SET_MAX_AVAILABILITY:
        raise ValueError(f"Unknown feature set: {feature_set}")
    allowed = FEATURE_SET_MAX_AVAILABILITY[feature_set]
    return [column for column in columns if feature_info(column).availability in allowed]


def leakage_report(columns: list[str], feature_set: str) -> list[dict[str, str]]:
    allowed = FEATURE_SET_MAX_AVAILABILITY[feature_set]
    rows = []
    for column in columns:
        info = feature_info(column)
        rows.append(
            {
                "column": column,
                "availability": info.availability,
                "family": info.family,
                "allowed_by_feature_set": str(info.availability in allowed).lower(),
                "description": info.description,
            }
        )
    return rows
