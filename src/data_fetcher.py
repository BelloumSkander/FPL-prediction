"""Fetch and cache data from the official FPL API."""

import json
import os
import time
from datetime import datetime, timedelta

import requests

from config import (
    BOOTSTRAP_URL,
    CACHE_EXPIRY_HOURS,
    DATA_DIR,
    FIXTURES_URL,
    PLAYER_SUMMARY_URL,
    PLAYERS_CACHE_DIR,
)


def _is_cache_valid(filepath: str) -> bool:
    """Check if a cached file exists and is within the expiry window."""
    if not os.path.exists(filepath):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now() - mtime < timedelta(hours=CACHE_EXPIRY_HOURS)


def _fetch_json(url: str, retries: int = 3) -> dict:
    """GET request with retry logic and exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {e}")
            time.sleep(2 ** attempt)


def fetch_bootstrap_static(force_refresh: bool = False) -> dict:
    """Fetch the bootstrap-static endpoint (all players, teams, gameweeks).

    Returns dict with keys: 'elements', 'teams', 'events', 'element_types'.
    """
    filepath = os.path.join(DATA_DIR, "bootstrap_static.json")
    if not force_refresh and _is_cache_valid(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    data = _fetch_json(BOOTSTRAP_URL)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_fixtures(force_refresh: bool = False) -> list:
    """Fetch all fixtures for the season."""
    filepath = os.path.join(DATA_DIR, "fixtures.json")
    if not force_refresh and _is_cache_valid(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    data = _fetch_json(FIXTURES_URL)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_player_history(player_id: int, force_refresh: bool = False) -> dict:
    """Fetch individual player gameweek history.

    Returns dict with keys: 'history', 'fixtures', 'history_past'.
    """
    os.makedirs(PLAYERS_CACHE_DIR, exist_ok=True)
    filepath = os.path.join(PLAYERS_CACHE_DIR, f"player_{player_id}.json")

    if not force_refresh and _is_cache_valid(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    url = PLAYER_SUMMARY_URL.format(player_id=player_id)
    data = _fetch_json(url)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_all_player_histories(
    player_ids: list,
    force_refresh: bool = False,
    progress_callback=None,
) -> dict:
    """Fetch history for all players with rate limiting.

    Args:
        player_ids: list of player IDs to fetch.
        force_refresh: bypass cache.
        progress_callback: callable(current, total) for progress updates.

    Returns dict mapping player_id -> history dict.
    """
    all_histories = {}
    for i, pid in enumerate(player_ids):
        try:
            all_histories[pid] = fetch_player_history(pid, force_refresh)
        except RuntimeError:
            # Skip players whose data can't be fetched
            continue
        if progress_callback:
            progress_callback(i + 1, len(player_ids))
        # Rate limit: avoid hitting the API too fast
        if force_refresh or not _is_cache_valid(
            os.path.join(PLAYERS_CACHE_DIR, f"player_{pid}.json")
        ):
            time.sleep(0.5)
    return all_histories


def get_current_gameweek(bootstrap_data: dict) -> int:
    """Determine the current gameweek from events data."""
    for event in bootstrap_data["events"]:
        if event["is_current"]:
            return event["id"]
    # Fallback: last finished event
    finished = [e for e in bootstrap_data["events"] if e["finished"]]
    return finished[-1]["id"] if finished else 1


def get_next_gameweek(bootstrap_data: dict) -> int:
    """Determine the next gameweek to predict for."""
    for event in bootstrap_data["events"]:
        if event["is_next"]:
            return event["id"]
    return get_current_gameweek(bootstrap_data) + 1
