from __future__ import annotations

import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .constants import CRICSHEET_IPL_JSON_URL, IPL_COMPETITION_URL, TEAM_ALIASES, VENUE_ALIASES


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, force: bool = False, timeout: int = 60) -> Path:
    ensure_parent(destination)
    if destination.exists() and not force:
        return destination
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_cricsheet_archive(raw_dir: Path, force: bool = False) -> Path:
    archive_path = raw_dir / "ipl_json.zip"
    return download_file(CRICSHEET_IPL_JSON_URL, archive_path, force=force)


def extract_zip(archive_path: Path, destination: Path, force: bool = False) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted"
    if marker.exists() and not force:
        return destination
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination)
    marker.write_text("ok\n")
    return destination


def parse_jsonp(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    match = re.match(r"^[^(]+\((.*)\)\s*;?\s*$", payload, flags=re.DOTALL)
    if not match:
        raise ValueError("Unable to parse JSONP payload.")
    return json.loads(match.group(1))


def download_official_schedule(raw_dir: Path, season: int, force: bool = False) -> Path:
    competition_path = raw_dir / "official" / "competition.js"
    schedule_path = raw_dir / "official" / f"{season}_schedule.js"
    download_file(IPL_COMPETITION_URL, competition_path, force=force)
    competition_data = parse_jsonp(competition_path.read_text())
    competition = next(
        item for item in competition_data["competition"] if str(item["SeasonID"]) == str(season - 2007)
    )
    schedule_url = f"{competition['feedsource']}/{competition['CompetitionID']}-matchschedule.js"
    return download_file(schedule_url, schedule_path, force=force)


def canonical_team(team_name: str) -> str:
    return TEAM_ALIASES.get(team_name, team_name)


def canonical_venue(venue_name: str) -> str:
    return VENUE_ALIASES.get(venue_name, venue_name)


def season_from_info(match_info: dict[str, Any]) -> int:
    dates = match_info.get("dates") or []
    if dates:
        return int(str(dates[0])[:4])
    season = str(match_info.get("season", ""))
    if "/" in season:
        first, second = season.split("/")
        second = second if len(second) == 4 else f"{first[:2]}{second}"
        return int(second)
    return int(season)


def _phase_for_over(over_number: int) -> str:
    if over_number < 6:
        return "pp"
    if over_number < 15:
        return "mid"
    return "death"


def _new_batting_record() -> dict[str, Any]:
    return {
        "runs": 0,
        "balls": 0,
        "fours": 0,
        "sixes": 0,
        "dismissed": 0,
        "phase_runs_pp": 0,
        "phase_runs_mid": 0,
        "phase_runs_death": 0,
    }


def _new_bowling_record() -> dict[str, Any]:
    return {
        "balls": 0,
        "runs_conceded": 0,
        "wickets": 0,
        "dot_balls": 0,
        "phase_runs_pp": 0,
        "phase_runs_mid": 0,
        "phase_runs_death": 0,
        "phase_balls_pp": 0,
        "phase_balls_mid": 0,
        "phase_balls_death": 0,
    }


def _overs_from_balls(balls: int) -> float:
    return math.floor(balls / 6) + (balls % 6) / 10.0


def parse_match_file(match_path: Path) -> dict[str, Any]:
    raw = json.loads(match_path.read_text())
    info = raw["info"]
    registry = (info.get("registry") or {}).get("people", {})
    teams = [canonical_team(team) for team in info.get("teams", [])]
    innings_map: dict[str, dict[str, Any]] = {}
    lineup_map: dict[str, list[dict[str, str]]] = {}

    for team_name, players in (info.get("players") or {}).items():
        canonical_name = canonical_team(team_name)
        lineup_map[canonical_name] = [
            {"player_id": registry.get(name, name), "player_name": name}
            for name in players
        ]

    for innings in raw.get("innings", []):
        team_name = canonical_team(innings["team"])
        batting = defaultdict(_new_batting_record)
        bowling = defaultdict(_new_bowling_record)
        batting_names: dict[str, str] = {}
        bowling_names: dict[str, str] = {}
        batting_order: list[str] = []
        wickets_total = 0
        total_runs = 0
        total_balls = 0
        powerplay_runs = 0
        powerplay_wickets = 0

        for over in innings.get("overs", []):
            over_number = over.get("over", 0)
            phase = _phase_for_over(over_number)
            for delivery in over.get("deliveries", []):
                batter_name = delivery["batter"]
                bowler_name = delivery["bowler"]
                batter_id = registry.get(batter_name, batter_name)
                bowler_id = registry.get(bowler_name, bowler_name)
                batting_names[batter_id] = batter_name
                bowling_names[bowler_id] = bowler_name
                if batter_id not in batting_order:
                    batting_order.append(batter_id)
                non_striker = delivery.get("non_striker")
                if non_striker:
                    non_striker_id = registry.get(non_striker, non_striker)
                    batting_names[non_striker_id] = non_striker
                    if non_striker_id not in batting_order:
                        batting_order.append(non_striker_id)

                extras_detail = delivery.get("extras") or {}
                batter_runs = int(delivery["runs"]["batter"])
                total = int(delivery["runs"]["total"])
                wides = int(extras_detail.get("wides", 0))
                noballs = int(extras_detail.get("noballs", 0))
                byes = int(extras_detail.get("byes", 0))
                legbyes = int(extras_detail.get("legbyes", 0))

                batting[batter_id]["runs"] += batter_runs
                if wides == 0:
                    batting[batter_id]["balls"] += 1
                if batter_runs == 4:
                    batting[batter_id]["fours"] += 1
                if batter_runs == 6:
                    batting[batter_id]["sixes"] += 1
                batting[batter_id][f"phase_runs_{phase}"] += batter_runs

                legal_ball = 1 if wides == 0 and noballs == 0 else 0
                if legal_ball:
                    total_balls += 1
                    bowling[bowler_id]["balls"] += 1
                    bowling[bowler_id][f"phase_balls_{phase}"] += 1
                conceded = batter_runs + wides + noballs
                bowling[bowler_id]["runs_conceded"] += conceded
                bowling[bowler_id][f"phase_runs_{phase}"] += conceded
                if legal_ball and total == 0:
                    bowling[bowler_id]["dot_balls"] += 1

                total_runs += total
                if over_number < 6:
                    powerplay_runs += total

                for wicket in delivery.get("wickets", []):
                    wickets_total += 1
                    out_name = wicket["player_out"]
                    out_id = registry.get(out_name, out_name)
                    batting_names[out_id] = out_name
                    batting[out_id]["dismissed"] = 1
                    credited_kinds = {
                        "bowled",
                        "caught",
                        "caught and bowled",
                        "lbw",
                        "stumped",
                        "hit wicket",
                    }
                    if wicket.get("kind") in credited_kinds:
                        bowling[bowler_id]["wickets"] += 1
                    if over_number < 6:
                        powerplay_wickets += 1

        innings_map[team_name] = {
            "batting": [
                {
                    "player_id": player_id,
                    "player_name": batting_names[player_id],
                    **stats,
                }
                for player_id, stats in batting.items()
            ],
            "bowling": [
                {
                    "player_id": player_id,
                    "player_name": bowling_names[player_id],
                    **stats,
                    "overs": _overs_from_balls(stats["balls"]),
                    "economy": (stats["runs_conceded"] * 6 / stats["balls"]) if stats["balls"] else 0.0,
                    "dot_ball_pct": (stats["dot_balls"] / stats["balls"]) if stats["balls"] else 0.0,
                }
                for player_id, stats in bowling.items()
            ],
            "batting_order": batting_order,
            "runs": total_runs,
            "wickets": wickets_total,
            "balls": total_balls,
            "overs": _overs_from_balls(total_balls),
            "powerplay_runs": powerplay_runs,
            "powerplay_wickets": powerplay_wickets,
        }

    outcome = info.get("outcome", {})
    match_number = (info.get("event") or {}).get("match_number")
    stage = (info.get("event") or {}).get("stage")

    result_type = "completed"
    if outcome.get("result") == "no result":
        result_type = "no_result"
    elif outcome.get("winner") is None:
        result_type = "tie"

    winner = canonical_team(outcome["winner"]) if outcome.get("winner") else None
    toss = info.get("toss") or {}
    toss_winner = canonical_team(toss["winner"]) if toss.get("winner") else None

    return {
        "match_id": match_path.stem,
        "season": season_from_info(info),
        "date": pd.to_datetime(info["dates"][0]),
        "venue": canonical_venue(info.get("venue", "Unknown Venue")),
        "city": info.get("city"),
        "team_a": teams[0],
        "team_b": teams[1],
        "winner": winner,
        "result_type": result_type,
        "toss_winner": toss_winner,
        "toss_decision": toss.get("decision"),
        "player_of_match": (info.get("player_of_match") or [None])[0],
        "match_number": int(match_number) if match_number else None,
        "stage": stage,
        "is_playoff": bool(stage),
        "lineups": lineup_map,
        "innings_map": innings_map,
    }


def load_cricsheet_matches(extracted_dir: Path) -> list[dict[str, Any]]:
    matches = []
    for match_path in sorted(extracted_dir.glob("*.json")):
        matches.append(parse_match_file(match_path))
    matches.sort(key=lambda match: (match["date"], match["match_id"]))
    return matches


def schedule_frame(schedule_path: Path) -> pd.DataFrame:
    raw = parse_jsonp(schedule_path.read_text())
    frame = pd.DataFrame(raw["Matchsummary"]).copy()
    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"])
    frame["team_a"] = frame["FirstBattingTeamName"].map(canonical_team)
    frame["team_b"] = frame["SecondBattingTeamName"].map(canonical_team)
    frame["venue"] = frame["GroundName"].map(canonical_venue)
    frame["season"] = frame["MatchDate"].dt.year
    frame["match_number"] = frame["MatchOrder"].str.extract(r"(\d+)").astype(float)
    frame = frame.rename(
        columns={
            "MatchID": "match_id",
            "MatchStatus": "match_status",
            "MatchDate": "date",
            "city": "city",
        }
    )
    return frame[
        [
            "match_id",
            "date",
            "season",
            "match_status",
            "team_a",
            "team_b",
            "venue",
            "city",
            "match_number",
        ]
    ].sort_values(["date", "match_number", "match_id"])


def build_match_table(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for match in matches:
        innings_a = match["innings_map"].get(match["team_a"], {})
        innings_b = match["innings_map"].get(match["team_b"], {})
        rows.append(
            {
                "match_id": match["match_id"],
                "season": match["season"],
                "date": match["date"],
                "venue": match["venue"],
                "city": match["city"],
                "team_a": match["team_a"],
                "team_b": match["team_b"],
                "winner": match["winner"],
                "result_type": match["result_type"],
                "toss_winner": match["toss_winner"],
                "toss_decision": match["toss_decision"],
                "is_playoff": match["is_playoff"],
                "team_a_runs": innings_a.get("runs"),
                "team_a_wickets": innings_a.get("wickets"),
                "team_b_runs": innings_b.get("runs"),
                "team_b_wickets": innings_b.get("wickets"),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "match_id"])

