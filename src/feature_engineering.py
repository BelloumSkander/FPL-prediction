"""Transform raw FPL API data into ML-ready feature DataFrames."""

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    FORM_WINDOW,
    MIN_MINUTES_THRESHOLD,
    PLAYER_PHOTO_URL,
    POSITION_MAP,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
)


def build_player_base_df(bootstrap_data: dict) -> pd.DataFrame:
    """Create a base DataFrame from bootstrap-static player elements."""
    players = bootstrap_data["elements"]
    df = pd.DataFrame(players)

    # Keep relevant columns and convert types
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["now_cost"] = df["now_cost"] / 10.0
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce").fillna(0)

    for col in ["influence", "creativity", "threat", "ict_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def build_team_lookup(bootstrap_data: dict) -> dict:
    """Create a mapping of team_id -> team info dict."""
    return {t["id"]: t for t in bootstrap_data["teams"]}


def build_fixture_df(fixtures_data: list) -> pd.DataFrame:
    """Convert fixtures list to a DataFrame."""
    df = pd.DataFrame(fixtures_data)
    return df


def compute_rolling_features(player_history: list, windows: list = None) -> dict:
    """Compute rolling statistics from a player's gameweek history.

    Only includes gameweeks where the player had >= MIN_MINUTES_THRESHOLD minutes.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    qualifying = [gw for gw in player_history if gw["minutes"] >= MIN_MINUTES_THRESHOLD]

    features = {}
    for w in windows:
        recent = qualifying[-w:] if len(qualifying) >= w else qualifying
        if not recent:
            for stat in ["points", "minutes", "goals", "assists", "clean_sheets", "bonus"]:
                features[f"rolling_{w}gw_{stat}"] = 0.0
            continue

        features[f"rolling_{w}gw_points"] = np.mean([g["total_points"] for g in recent])
        features[f"rolling_{w}gw_minutes"] = np.mean([g["minutes"] for g in recent])
        features[f"rolling_{w}gw_goals"] = sum(g["goals_scored"] for g in recent)
        features[f"rolling_{w}gw_assists"] = sum(g["assists"] for g in recent)
        features[f"rolling_{w}gw_clean_sheets"] = sum(g["clean_sheets"] for g in recent)
        features[f"rolling_{w}gw_bonus"] = sum(g["bonus"] for g in recent)

    return features


def compute_minutes_trend(player_history: list, window: int = 5) -> float:
    """Calculate the linear slope of minutes played over the last N gameweeks."""
    recent = player_history[-window:]
    if len(recent) < 2:
        return 0.0
    minutes_series = [gw["minutes"] for gw in recent]
    x = np.arange(len(minutes_series))
    slope, _ = np.polyfit(x, minutes_series, 1)
    return float(slope)


def compute_per90_stats(player_history: list) -> dict:
    """Calculate goals and assists per 90 minutes."""
    total_minutes = sum(gw["minutes"] for gw in player_history)
    if total_minutes < 90:
        return {"goals_per_90": 0.0, "assists_per_90": 0.0}

    total_goals = sum(gw["goals_scored"] for gw in player_history)
    total_assists = sum(gw["assists"] for gw in player_history)

    return {
        "goals_per_90": (total_goals / total_minutes) * 90,
        "assists_per_90": (total_assists / total_minutes) * 90,
    }


def compute_clean_sheet_probability(player_history: list, element_type: int) -> float:
    """Estimate clean sheet probability for GK/DEF players."""
    if element_type not in [1, 2]:
        return 0.0
    qualifying = [gw for gw in player_history if gw["minutes"] >= 60]
    if not qualifying:
        return 0.0
    cs_count = sum(1 for gw in qualifying if gw["clean_sheets"] > 0)
    return cs_count / len(qualifying)


def compute_xg_overperformance(player_history: list) -> float:
    """Calculate goals_scored minus expected_goals over the season."""
    total_goals = sum(gw["goals_scored"] for gw in player_history)
    total_xg = sum(float(gw.get("expected_goals", 0) or 0) for gw in player_history)
    return total_goals - total_xg


def get_next_opponent_difficulty(
    player_team: int, next_gw: int, fixtures_df: pd.DataFrame
) -> tuple:
    """Get the FDR and home/away status for the player's next fixture.

    Returns (opponent_difficulty, is_home).
    """
    gw_fixtures = fixtures_df[fixtures_df["event"] == next_gw]

    # Check if the player's team is playing at home
    home_match = gw_fixtures[gw_fixtures["team_h"] == player_team]
    if not home_match.empty:
        return int(home_match.iloc[0].get("team_h_difficulty", 3)), 1

    # Check if the player's team is playing away
    away_match = gw_fixtures[gw_fixtures["team_a"] == player_team]
    if not away_match.empty:
        return int(away_match.iloc[0].get("team_a_difficulty", 3)), 0

    # No fixture found (blank gameweek)
    return 3, 0


def get_next_opponent_info(
    player_team: int, next_gw: int, fixtures_df: pd.DataFrame, team_lookup: dict
) -> dict:
    """Get detailed next fixture info including opponent name and venue.

    Returns dict with opponent_name, opponent_short, difficulty, is_home.
    """
    gw_fixtures = fixtures_df[fixtures_df["event"] == next_gw]

    home_match = gw_fixtures[gw_fixtures["team_h"] == player_team]
    if not home_match.empty:
        opp_id = int(home_match.iloc[0]["team_a"])
        opp_info = team_lookup.get(opp_id, {})
        return {
            "opponent_name": opp_info.get("name", "Unknown"),
            "opponent_short": opp_info.get("short_name", "???"),
            "difficulty": int(home_match.iloc[0].get("team_h_difficulty", 3)),
            "is_home": True,
        }

    away_match = gw_fixtures[gw_fixtures["team_a"] == player_team]
    if not away_match.empty:
        opp_id = int(away_match.iloc[0]["team_h"])
        opp_info = team_lookup.get(opp_id, {})
        return {
            "opponent_name": opp_info.get("name", "Unknown"),
            "opponent_short": opp_info.get("short_name", "???"),
            "difficulty": int(away_match.iloc[0].get("team_a_difficulty", 3)),
            "is_home": False,
        }

    return {
        "opponent_name": "Blank GW",
        "opponent_short": "BGW",
        "difficulty": 3,
        "is_home": False,
    }


def _build_row_from_history(
    player_info: pd.Series,
    prior_history: list,
    target_gw_data: dict,
    fixtures_df: pd.DataFrame,
    include_target: bool = True,
) -> dict:
    """Build a single feature row from player info and history.

    Shared logic between training and prediction feature building.
    """
    rolling = compute_rolling_features(prior_history)
    per90 = compute_per90_stats(prior_history)
    minutes_trend = compute_minutes_trend(prior_history)
    cs_prob = compute_clean_sheet_probability(prior_history, player_info["element_type"])
    xg_over = compute_xg_overperformance(prior_history)

    # ICT averages from last 5 qualifying gameweeks
    recent_5 = [
        g for g in prior_history if g["minutes"] >= MIN_MINUTES_THRESHOLD
    ][-5:]

    def _safe_mean(records, key):
        if not records:
            return 0.0
        return float(np.mean([float(g.get(key, 0) or 0) for g in records]))

    row = {
        "player_id": player_info["id"],
        "web_name": player_info["web_name"],
        **rolling,
        "opponent_difficulty": target_gw_data.get("difficulty", 3),
        "is_home": int(target_gw_data.get("was_home", False)),
        "position_encoded": player_info["element_type"],
        "now_cost": float(player_info["now_cost"]),
        "selected_by_percent": float(player_info["selected_by_percent"]),
        "influence": _safe_mean(recent_5, "influence"),
        "creativity": _safe_mean(recent_5, "creativity"),
        "threat": _safe_mean(recent_5, "threat"),
        "ict_index": _safe_mean(recent_5, "ict_index"),
        "goals_per_90": per90["goals_per_90"],
        "assists_per_90": per90["assists_per_90"],
        "minutes_trend": minutes_trend,
        "clean_sheet_probability": cs_prob,
        "xg_overperformance": xg_over,
    }

    if include_target and target_gw_data is not None:
        row[TARGET_COLUMN] = target_gw_data.get("total_points", 0)

    return row


def build_training_dataset(
    bootstrap_data: dict,
    all_histories: dict,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full training dataset from historical gameweek data.

    Each row is a (player, gameweek) pair. Features only use data BEFORE
    the target gameweek to prevent data leakage.
    """
    players_df = build_player_base_df(bootstrap_data)
    rows = []

    for player_id, history_data in all_histories.items():
        history = history_data.get("history", [])
        if len(history) < FORM_WINDOW + 1:
            continue

        player_rows = players_df[players_df["id"] == player_id]
        if player_rows.empty:
            continue
        player_info = player_rows.iloc[0]

        for i in range(FORM_WINDOW, len(history)):
            target_gw = history[i]
            prior_history = history[:i]

            row = _build_row_from_history(
                player_info, prior_history, target_gw, fixtures_df, include_target=True
            )
            row["gameweek"] = target_gw["round"]
            rows.append(row)

    return pd.DataFrame(rows)


def build_prediction_features(
    bootstrap_data: dict,
    all_histories: dict,
    fixtures_df: pd.DataFrame,
    next_gw: int,
) -> pd.DataFrame:
    """Build feature vectors for the next gameweek (no target).

    One row per available player with sufficient history.
    """
    players_df = build_player_base_df(bootstrap_data)
    team_lookup = build_team_lookup(bootstrap_data)
    rows = []

    for player_id, history_data in all_histories.items():
        history = history_data.get("history", [])
        if len(history) < FORM_WINDOW:
            continue

        player_rows = players_df[players_df["id"] == player_id]
        if player_rows.empty:
            continue
        player_info = player_rows.iloc[0]

        # Skip only fully unavailable players (released/loaned)
        if player_info.get("status") == "u":
            continue

        # Get next fixture info
        opp_info = get_next_opponent_info(
            int(player_info["team"]), next_gw, fixtures_df, team_lookup
        )

        # Build a synthetic target_gw_data dict for the shared row builder
        target_gw_data = {
            "difficulty": opp_info["difficulty"],
            "was_home": opp_info["is_home"],
        }

        row = _build_row_from_history(
            player_info, history, target_gw_data, fixtures_df, include_target=False
        )
        row["team"] = team_lookup.get(int(player_info["team"]), {}).get("short_name", "???")
        row["team_full"] = team_lookup.get(int(player_info["team"]), {}).get("name", "Unknown")
        row["position"] = player_info["position"]

        # Player photo URL
        code = player_info.get("code", "")
        row["photo_url"] = PLAYER_PHOTO_URL.format(code=code)

        # Full name for display
        row["first_name"] = player_info.get("first_name", "")
        row["second_name"] = player_info.get("second_name", "")

        # Next match display data
        row["next_opponent"] = opp_info["opponent_short"]
        row["next_opponent_full"] = opp_info["opponent_name"]
        row["next_difficulty"] = opp_info["difficulty"]
        row["next_is_home"] = opp_info["is_home"]

        # Current season stats from bootstrap
        row["total_points_season"] = int(player_info.get("total_points", 0))
        row["goals_season"] = int(player_info.get("goals_scored", 0))
        row["assists_season"] = int(player_info.get("assists_scored", player_info.get("assists", 0)))
        row["minutes_season"] = int(player_info.get("minutes", 0))
        row["form_value"] = float(player_info.get("form", 0))
        row["points_per_game_value"] = float(player_info.get("points_per_game", 0))
        row["status"] = player_info.get("status", "a")
        row["news"] = player_info.get("news", "")
        row["chance_next_round"] = player_info.get("chance_of_playing_next_round")

        # Previous season data
        history_past = history_data.get("history_past", [])
        if history_past:
            last_season = history_past[-1]
            row["prev_season_name"] = last_season.get("season_name", "N/A")
            row["prev_season_points"] = int(last_season.get("total_points", 0))
            row["prev_season_minutes"] = int(last_season.get("minutes", 0))
            row["prev_season_goals"] = int(last_season.get("goals_scored", 0))
            row["prev_season_assists"] = int(last_season.get("assists", 0))
            row["prev_season_cs"] = int(last_season.get("clean_sheets", 0))
        else:
            row["prev_season_name"] = "N/A"
            row["prev_season_points"] = 0
            row["prev_season_minutes"] = 0
            row["prev_season_goals"] = 0
            row["prev_season_assists"] = 0
            row["prev_season_cs"] = 0

        rows.append(row)

    return pd.DataFrame(rows)
