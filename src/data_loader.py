"""
data_loader.py — Loading functions for the two raw data sources.
Master Scout · Stage 2.

Functions:
    load_players_data(path, fifa_version=24)  -> FC24 player profiles
    load_events_data(path)                    -> Football Events match events
"""

from pathlib import Path
import pandas as pd

# ---- Column selections (decided in Stage 1, docs/01_schema.md) ----

PLAYER_COLUMNS = [
    # identity / meta
    "player_id", "fifa_version", "short_name", "long_name", "player_positions",
    "age", "dob", "nationality_name", "club_name", "league_name", "preferred_foot",
    # market / value
    "value_eur", "wage_eur", "release_clause_eur", "overall", "potential",
    "international_reputation",
    # physical
    "height_cm", "weight_kg", "weak_foot", "skill_moves", "work_rate", "body_type",
    # core 6 face stats
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    # selected detailed attributes
    "attacking_finishing", "attacking_short_passing", "attacking_crossing",
    "attacking_heading_accuracy", "skill_dribbling", "skill_ball_control",
    "skill_long_passing", "skill_fk_accuracy", "movement_acceleration",
    "movement_sprint_speed", "movement_agility", "movement_reactions",
    "power_shot_power", "power_stamina", "power_strength", "power_long_shots",
    "mentality_vision", "mentality_positioning", "mentality_interceptions",
    "mentality_composure", "defending_standing_tackle", "defending_sliding_tackle",
    "defending_marking_awareness",
]

EVENT_COLUMNS = [
    "id_odsp", "time", "text", "event_type", "event_type2", "side",
    "event_team", "opponent", "player", "player2", "shot_place", "shot_outcome",
    "is_goal", "location", "bodypart", "assist_method", "situation", "fast_break",
]


def load_players_data(path, fifa_version=24):
    """Load FC player profiles. By default keeps only the latest edition (FC24).

    Args:
        path: path to male_players.csv
        fifa_version: which FIFA edition to keep (default 24). Pass None to keep all.

    Returns:
        DataFrame, one row per player (for the chosen version).
    """
    df = pd.read_csv(path, low_memory=False, usecols=PLAYER_COLUMNS)
    if fifa_version is not None:
        df = df[df["fifa_version"] == fifa_version].copy()
        # within one edition a player_id is unique; guard anyway
        df = df.drop_duplicates(subset="player_id", keep="first").reset_index(drop=True)
    return df


def load_events_data(path):
    """Load Football Events match events.

    Args:
        path: path to events.csv

    Returns:
        DataFrame, one row per match event.
    """
    df = pd.read_csv(path, low_memory=False, usecols=EVENT_COLUMNS)
    return df


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    players = load_players_data(here / "data/raw/male_players.csv")
    events = load_events_data(here / "data/raw/events.csv")
    print("players (FC24):", players.shape)
    print("events:", events.shape)
