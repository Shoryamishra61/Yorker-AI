from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import CURRENT_SEASON, TEAM_HOME_VENUES, VENUE_METADATA


def safe_mean(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def last_values(items: list[dict[str, Any]], key: str, n: int) -> list[float]:
    return [float(item[key]) for item in items[-n:] if item.get(key) is not None]


def _normalize(value: float, scale: float, cap: float = 2.0) -> float:
    return max(0.0, min(value / scale, cap))


def _inverse_normalize(value: float, scale: float, cap: float = 2.0) -> float:
    return max(0.0, min((scale - value) / scale, cap))


def _elo_probability(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))


@dataclass
class TeamSnapshot:
    team_elo: float
    season_win_rate: float
    season_nrr: float
    season_runs_for: float
    season_runs_against: float
    season_matches_played: float
    current_streak: float
    days_since_last_match: float
    recent_win_rate_5: float
    historical_win_rate: float
    recent_nrr_5: float
    recent_runs_for_5: float
    recent_runs_against_5: float
    powerplay_runs_5: float
    powerplay_wickets_5: float
    venue_win_rate: float
    head_to_head_win_rate: float
    batting_power: float
    bowling_danger: float
    xi_experience: float
    xi_batting_form: float
    xi_bowling_form: float
    xi_batting_vs_opponent: float
    xi_bowling_vs_opponent: float
    home_advantage: float
    known_xi_ratio: float


class FeatureBuilder:
    def __init__(self) -> None:
        self.team_history = defaultdict(list)
        self.team_venue_history = defaultdict(list)
        self.head_to_head = defaultdict(list)
        self.player_batting_history = defaultdict(list)
        self.player_bowling_history = defaultdict(list)
        self.player_batting_vs_team = defaultdict(list)
        self.player_bowling_vs_team = defaultdict(list)
        self.player_batting_pos = defaultdict(list)
        self.player_match_counts = Counter()
        self.player_display_names: dict[str, str] = {}
        self.team_lineups = defaultdict(list)
        self.team_elos = defaultdict(lambda: 1500.0)
        self.dataset_rows: list[dict[str, Any]] = []
        self.current_match_table_rows: list[dict[str, Any]] = []

    def _batting_rating(self, player_id: str, venue: str, opponent: str) -> tuple[float, float]:
        history = self.player_batting_history[player_id]
        recent = history[-5:]
        venue_history = [item for item in history if item["venue"] == venue]
        opp_history = self.player_batting_vs_team[(player_id, opponent)][-5:]
        recent_runs = safe_mean([item["runs"] for item in recent], 20.0)
        recent_sr = safe_mean(
            [item["runs"] * 100 / item["balls"] for item in recent if item["balls"] > 0],
            110.0,
        )
        venue_runs = safe_mean([item["runs"] for item in venue_history[-5:]], recent_runs)
        opp_runs = safe_mean([item["runs"] for item in opp_history], recent_runs)
        form = 0.45 * _normalize(recent_runs, 35.0) + 0.35 * _normalize(recent_sr, 145.0) + 0.20 * _normalize(
            venue_runs, 35.0
        )
        matchup = 0.6 * _normalize(opp_runs, 35.0) + 0.4 * _normalize(venue_runs, 35.0)
        return form, matchup

    def _bowling_rating(self, player_id: str, venue: str, opponent: str) -> tuple[float, float]:
        history = self.player_bowling_history[player_id]
        recent = history[-5:]
        venue_history = [item for item in history if item["venue"] == venue]
        opp_history = self.player_bowling_vs_team[(player_id, opponent)][-5:]
        recent_wkts = safe_mean([item["wickets"] for item in recent], 0.4)
        recent_econ = safe_mean([item["economy"] for item in recent if item["balls"] > 0], 8.8)
        venue_econ = safe_mean([item["economy"] for item in venue_history[-5:] if item["balls"] > 0], recent_econ)
        opp_wkts = safe_mean([item["wickets"] for item in opp_history], recent_wkts)
        form = 0.5 * _normalize(recent_wkts, 1.5) + 0.3 * _inverse_normalize(recent_econ, 10.0) + 0.2 * _inverse_normalize(
            venue_econ, 10.0
        )
        matchup = 0.6 * _normalize(opp_wkts, 1.5) + 0.4 * _inverse_normalize(venue_econ, 10.0)
        return form, matchup

    def _estimate_batting_order(self, lineup: list[dict[str, str]]) -> list[dict[str, str]]:
        def sort_key(player: dict[str, str]) -> tuple[float, float]:
            positions = self.player_batting_pos[player["player_id"]]
            avg_position = safe_mean(positions, 7.5)
            role_penalty = 0.0 if self.player_batting_history[player["player_id"]] else 2.0
            return (avg_position + role_penalty, -self.player_match_counts[player["player_id"]])

        return sorted(lineup, key=sort_key)

    def _estimate_bowling_group(self, lineup: list[dict[str, str]]) -> list[dict[str, str]]:
        return sorted(
            lineup,
            key=lambda player: (
                -len(self.player_bowling_history[player["player_id"]]),
                -self.player_match_counts[player["player_id"]],
            ),
        )[:6]

    def _current_streak(self, team_history: list[dict[str, Any]]) -> float:
        if not team_history:
            return 0.0
        latest_result = int(team_history[-1]["win"])
        direction = 1.0 if latest_result == 1 else -1.0
        streak = 0
        for item in reversed(team_history):
            if int(item["win"]) != latest_result:
                break
            streak += 1
        return direction * float(streak)

    def _team_snapshot(
        self,
        team: str,
        lineup: list[dict[str, str]],
        venue: str,
        opponent: str,
        as_of_date: pd.Timestamp,
        season: int,
    ) -> TeamSnapshot:
        team_history = self.team_history[team]
        recent = team_history[-5:]
        season_history = [item for item in team_history if item["season"] == season]
        venue_history = self.team_venue_history[(team, venue)]
        h2h = self.head_to_head[(team, opponent)]
        sorted_batting = self._estimate_batting_order(lineup)[:7]
        sorted_bowling = self._estimate_bowling_group(lineup)

        batting_forms = []
        batting_matchups = []
        bowling_forms = []
        bowling_matchups = []
        experience = []

        for player in sorted_batting:
            player_id = player["player_id"]
            batting_form, batting_matchup = self._batting_rating(player_id, venue, opponent)
            batting_forms.append(batting_form)
            batting_matchups.append(batting_matchup)
            experience.append(self.player_match_counts[player_id])

        for player in sorted_bowling:
            player_id = player["player_id"]
            bowling_form, bowling_matchup = self._bowling_rating(player_id, venue, opponent)
            bowling_forms.append(bowling_form)
            bowling_matchups.append(bowling_matchup)
            experience.append(self.player_match_counts[player_id])

        last_match_date = team_history[-1]["date"] if team_history else None
        days_since_last_match = (
            float(max((pd.Timestamp(as_of_date) - pd.Timestamp(last_match_date)).days, 0))
            if last_match_date is not None
            else 14.0
        )

        return TeamSnapshot(
            team_elo=float(self.team_elos[team]),
            season_win_rate=safe_mean([item["win"] for item in season_history], 0.5),
            season_nrr=safe_mean([item["nrr"] for item in season_history], 0.0),
            season_runs_for=safe_mean([item["runs_for"] for item in season_history], 160.0),
            season_runs_against=safe_mean([item["runs_against"] for item in season_history], 160.0),
            season_matches_played=float(len(season_history)),
            current_streak=self._current_streak(team_history),
            days_since_last_match=days_since_last_match,
            recent_win_rate_5=safe_mean(last_values(recent, "win", 5), 0.5),
            historical_win_rate=safe_mean([item["win"] for item in team_history], 0.5),
            recent_nrr_5=safe_mean(last_values(recent, "nrr", 5), 0.0),
            recent_runs_for_5=safe_mean(last_values(recent, "runs_for", 5), 160.0),
            recent_runs_against_5=safe_mean(last_values(recent, "runs_against", 5), 160.0),
            powerplay_runs_5=safe_mean(last_values(recent, "powerplay_runs", 5), 45.0),
            powerplay_wickets_5=safe_mean(last_values(recent, "powerplay_wickets", 5), 1.5),
            venue_win_rate=safe_mean([item["win"] for item in venue_history], 0.5),
            head_to_head_win_rate=safe_mean(h2h, 0.5),
            batting_power=safe_mean(batting_forms, 0.5),
            bowling_danger=safe_mean(bowling_forms, 0.5),
            xi_experience=safe_mean(experience, 0.0),
            xi_batting_form=safe_mean(batting_forms, 0.5),
            xi_bowling_form=safe_mean(bowling_forms, 0.5),
            xi_batting_vs_opponent=safe_mean(batting_matchups, 0.5),
            xi_bowling_vs_opponent=safe_mean(bowling_matchups, 0.5),
            home_advantage=1.0 if venue in TEAM_HOME_VENUES.get(team, set()) else 0.0,
            known_xi_ratio=(sum(1 for player in lineup if self.player_match_counts[player["player_id"]] > 0) / max(len(lineup), 1)),
        )

    def _row_from_snapshots(
        self,
        match_id: str,
        date: pd.Timestamp,
        season: int,
        venue: str,
        is_playoff: bool,
        team_a: str,
        team_b: str,
        snapshot_a: TeamSnapshot,
        snapshot_b: TeamSnapshot,
        label: int | None,
    ) -> dict[str, Any]:
        venue_meta = VENUE_METADATA.get(venue, {})
        elo_diff = snapshot_a.team_elo - snapshot_b.team_elo
        home_adjusted_elo_diff = elo_diff + 35.0 * (snapshot_a.home_advantage - snapshot_b.home_advantage)
        return {
            "match_id": match_id,
            "date": date,
            "season": season,
            "venue": venue,
            "team_a": team_a,
            "team_b": team_b,
            "is_playoff": int(is_playoff),
            "month": int(date.month),
            "label": label,
            "elo_diff": elo_diff,
            "elo_expected_team_a": _elo_probability(elo_diff),
            "elo_home_expected_team_a": _elo_probability(home_adjusted_elo_diff),
            "season_win_rate_diff": snapshot_a.season_win_rate - snapshot_b.season_win_rate,
            "season_nrr_diff": snapshot_a.season_nrr - snapshot_b.season_nrr,
            "season_runs_for_diff": snapshot_a.season_runs_for - snapshot_b.season_runs_for,
            "season_runs_against_diff": snapshot_a.season_runs_against - snapshot_b.season_runs_against,
            "season_matches_played_diff": snapshot_a.season_matches_played - snapshot_b.season_matches_played,
            "current_streak_diff": snapshot_a.current_streak - snapshot_b.current_streak,
            "days_since_last_match_diff": snapshot_a.days_since_last_match - snapshot_b.days_since_last_match,
            "recent_win_rate_diff": snapshot_a.recent_win_rate_5 - snapshot_b.recent_win_rate_5,
            "historical_win_rate_diff": snapshot_a.historical_win_rate - snapshot_b.historical_win_rate,
            "recent_nrr_diff": snapshot_a.recent_nrr_5 - snapshot_b.recent_nrr_5,
            "recent_runs_for_diff": snapshot_a.recent_runs_for_5 - snapshot_b.recent_runs_for_5,
            "recent_runs_against_diff": snapshot_a.recent_runs_against_5 - snapshot_b.recent_runs_against_5,
            "powerplay_runs_diff": snapshot_a.powerplay_runs_5 - snapshot_b.powerplay_runs_5,
            "powerplay_wickets_diff": snapshot_a.powerplay_wickets_5 - snapshot_b.powerplay_wickets_5,
            "venue_win_rate_diff": snapshot_a.venue_win_rate - snapshot_b.venue_win_rate,
            "head_to_head_diff": snapshot_a.head_to_head_win_rate - snapshot_b.head_to_head_win_rate,
            "batting_power_diff": snapshot_a.batting_power - snapshot_b.batting_power,
            "bowling_danger_diff": snapshot_a.bowling_danger - snapshot_b.bowling_danger,
            "xi_experience_diff": snapshot_a.xi_experience - snapshot_b.xi_experience,
            "xi_batting_form_diff": snapshot_a.xi_batting_form - snapshot_b.xi_batting_form,
            "xi_bowling_form_diff": snapshot_a.xi_bowling_form - snapshot_b.xi_bowling_form,
            "xi_batting_vs_opponent_diff": snapshot_a.xi_batting_vs_opponent - snapshot_b.xi_batting_vs_opponent,
            "xi_bowling_vs_opponent_diff": snapshot_a.xi_bowling_vs_opponent - snapshot_b.xi_bowling_vs_opponent,
            "home_advantage_diff": snapshot_a.home_advantage - snapshot_b.home_advantage,
            "known_xi_ratio_diff": snapshot_a.known_xi_ratio - snapshot_b.known_xi_ratio,
            "venue_avg_first_innings_score": venue_meta.get("avg_first_innings_score", 170.0),
            "venue_chase_win_pct": venue_meta.get("chase_win_pct", 0.5),
            "venue_spin_bias": venue_meta.get("spin_bias", 0.45),
            "venue_pace_bias": venue_meta.get("pace_bias", 0.55),
        }

    def _update_team_history(self, match: dict[str, Any], team: str, opponent: str) -> None:
        innings_team = match["innings_map"][team]
        innings_opp = match["innings_map"][opponent]
        balls_for = max(innings_team["balls"], 1)
        balls_against = max(innings_opp["balls"], 1)
        nrr = (innings_team["runs"] * 6 / balls_for) - (innings_opp["runs"] * 6 / balls_against)
        team_record = {
            "date": match["date"],
            "season": match["season"],
            "opponent": opponent,
            "venue": match["venue"],
            "win": int(match["winner"] == team),
            "runs_for": innings_team["runs"],
            "runs_against": innings_opp["runs"],
            "powerplay_runs": innings_team["powerplay_runs"],
            "powerplay_wickets": innings_team["powerplay_wickets"],
            "nrr": nrr,
        }
        self.team_history[team].append(team_record)
        self.team_venue_history[(team, match["venue"])].append(team_record)
        self.head_to_head[(team, opponent)].append(team_record["win"])

    def _update_player_history(self, match: dict[str, Any], team: str, opponent: str) -> None:
        innings_team = match["innings_map"][team]
        batting_order = innings_team["batting_order"]
        order_index = {player_id: idx + 1 for idx, player_id in enumerate(batting_order)}

        for player in match["lineups"].get(team, []):
            self.player_display_names[player["player_id"]] = player["player_name"]
            self.player_match_counts[player["player_id"]] += 1

        for batting in innings_team["batting"]:
            player_id = batting["player_id"]
            record = {
                "date": match["date"],
                "season": match["season"],
                "team": team,
                "opponent": opponent,
                "venue": match["venue"],
                "runs": batting["runs"],
                "balls": batting["balls"],
                "dismissed": batting["dismissed"],
            }
            self.player_batting_history[player_id].append(record)
            self.player_batting_vs_team[(player_id, opponent)].append(record)
            self.player_batting_pos[player_id].append(order_index.get(player_id, 8))

        for bowling in innings_team["bowling"]:
            player_id = bowling["player_id"]
            record = {
                "date": match["date"],
                "season": match["season"],
                "team": team,
                "opponent": opponent,
                "venue": match["venue"],
                "wickets": bowling["wickets"],
                "economy": bowling["economy"],
                "balls": bowling["balls"],
            }
            self.player_bowling_history[player_id].append(record)
            self.player_bowling_vs_team[(player_id, opponent)].append(record)

        self.team_lineups[team].append(
            {
                "date": match["date"],
                "season": match["season"],
                "players": match["lineups"].get(team, []),
            }
        )

    def _update_team_elos(self, match: dict[str, Any]) -> None:
        if match["winner"] is None:
            return
        team_a = match["team_a"]
        team_b = match["team_b"]
        if team_a not in match["innings_map"] or team_b not in match["innings_map"]:
            return

        home_adjustment = 35.0 * (
            (1.0 if match["venue"] in TEAM_HOME_VENUES.get(team_a, set()) else 0.0)
            - (1.0 if match["venue"] in TEAM_HOME_VENUES.get(team_b, set()) else 0.0)
        )
        rating_diff = self.team_elos[team_a] - self.team_elos[team_b] + home_adjustment
        expected_a = _elo_probability(rating_diff)
        actual_a = 1.0 if match["winner"] == team_a else 0.0
        runs_a = match["innings_map"][team_a]["runs"]
        runs_b = match["innings_map"][team_b]["runs"]
        margin_multiplier = min(1.75, 1.0 + abs(runs_a - runs_b) / 80.0)
        update = 28.0 * margin_multiplier * (actual_a - expected_a)
        self.team_elos[team_a] += update
        self.team_elos[team_b] -= update

    def _update_histories(self, match: dict[str, Any]) -> None:
        if match["team_a"] not in match["innings_map"] or match["team_b"] not in match["innings_map"]:
            return
        self._update_team_history(match, match["team_a"], match["team_b"])
        self._update_team_history(match, match["team_b"], match["team_a"])
        self._update_player_history(match, match["team_a"], match["team_b"])
        self._update_player_history(match, match["team_b"], match["team_a"])
        self._update_team_elos(match)

    def build_dataset(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        self.dataset_rows = []
        for match in matches:
            team_a = match["team_a"]
            team_b = match["team_b"]
            lineup_a = match["lineups"].get(team_a, [])
            lineup_b = match["lineups"].get(team_b, [])
            if lineup_a and lineup_b and match["winner"] is not None:
                snapshot_a = self._team_snapshot(team_a, lineup_a, match["venue"], team_b, match["date"], match["season"])
                snapshot_b = self._team_snapshot(team_b, lineup_b, match["venue"], team_a, match["date"], match["season"])
                label = int(match["winner"] == team_a)
                self.dataset_rows.append(
                    self._row_from_snapshots(
                        match["match_id"],
                        match["date"],
                        match["season"],
                        match["venue"],
                        match["is_playoff"],
                        team_a,
                        team_b,
                        snapshot_a,
                        snapshot_b,
                        label,
                    )
                )
                self.dataset_rows.append(
                    self._row_from_snapshots(
                        match["match_id"],
                        match["date"],
                        match["season"],
                        match["venue"],
                        match["is_playoff"],
                        team_b,
                        team_a,
                        snapshot_b,
                        snapshot_a,
                        1 - label,
                    )
                )
            self._update_histories(match)
        return pd.DataFrame(self.dataset_rows).sort_values(["date", "match_id", "team_a"])

    def infer_probable_xi(self, team: str, as_of_date: pd.Timestamp, target_season: int = CURRENT_SEASON) -> list[dict[str, str]]:
        candidates = [
            lineup
            for lineup in self.team_lineups[team]
            if lineup["date"] < as_of_date and lineup["season"] == target_season
        ]
        if not candidates:
            candidates = [lineup for lineup in self.team_lineups[team] if lineup["date"] < as_of_date]
        if not candidates:
            return []
        player_scores = defaultdict(float)
        latest_names: dict[str, str] = {}
        for rank, lineup in enumerate(reversed(candidates[-5:]), start=1):
            weight = 1.0 / rank
            for player in lineup["players"]:
                player_scores[player["player_id"]] += 1.0 + weight
                latest_names[player["player_id"]] = player["player_name"]
        chosen = sorted(player_scores.items(), key=lambda item: item[1], reverse=True)[:11]
        return [{"player_id": player_id, "player_name": latest_names[player_id]} for player_id, _ in chosen]

    def upcoming_rows(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for fixture in fixtures.itertuples(index=False):
            lineup_a = self.infer_probable_xi(fixture.team_a, fixture.date, fixture.season)
            lineup_b = self.infer_probable_xi(fixture.team_b, fixture.date, fixture.season)
            if not lineup_a or not lineup_b:
                continue
            snapshot_a = self._team_snapshot(
                fixture.team_a,
                lineup_a,
                fixture.venue,
                fixture.team_b,
                fixture.date,
                int(fixture.season),
            )
            snapshot_b = self._team_snapshot(
                fixture.team_b,
                lineup_b,
                fixture.venue,
                fixture.team_a,
                fixture.date,
                int(fixture.season),
            )
            rows.append(
                self._row_from_snapshots(
                    str(fixture.match_id),
                    fixture.date,
                    int(fixture.season),
                    fixture.venue,
                    False,
                    fixture.team_a,
                    fixture.team_b,
                    snapshot_a,
                    snapshot_b,
                    None,
                )
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "match_id",
                    "date",
                    "season",
                    "venue",
                    "team_a",
                    "team_b",
                    "is_playoff",
                    "month",
                    "label",
                ]
            )
        return pd.DataFrame(rows).sort_values(["date", "match_id"])

    def current_table(self) -> pd.DataFrame:
        rows = []
        for team, matches in self.team_history.items():
            current = [match for match in matches if match["season"] == CURRENT_SEASON]
            if not current:
                continue
            wins = sum(item["win"] for item in current)
            losses = len(current) - wins
            runs_for = sum(item["runs_for"] for item in current)
            runs_against = sum(item["runs_against"] for item in current)
            nrr = safe_mean([item["nrr"] for item in current], 0.0)
            rows.append(
                {
                    "team": team,
                    "matches": len(current),
                    "wins": wins,
                    "losses": losses,
                    "points": wins * 2,
                    "nrr_proxy": nrr,
                    "runs_for": runs_for,
                    "runs_against": runs_against,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["points", "nrr_proxy"], ascending=[False, False]).reset_index(drop=True)
