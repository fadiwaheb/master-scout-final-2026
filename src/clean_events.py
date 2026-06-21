"""
clean_events.py — Stage 4.
Clean the Football Events table, translate codes to text names, and build the
binary feature columns used later for player aggregation.

Output: data/processed/clean_events.csv  (one row per match event)

Code mappings come from docs/01_schema.md (verified against the raw values).
The one numeric THRESHOLD here is the set of "inside the box" location codes
(is_box_shot) — documented in docs/ספים_והחלטות_Master_Scout.docx.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_players import clean_player_name  # reuse the same normalization key

# =====================================================================
# CODE -> NAME mappings (from docs/01_schema.md)
# =====================================================================
EVENT_TYPE_NAMES = {
    1: "Attempt", 2: "Corner", 3: "Foul", 4: "Yellow card",
    5: "Second yellow card", 6: "Red card", 7: "Substitution",
    8: "Free kick won", 9: "Offside", 10: "Hand ball", 11: "Penalty conceded",
}
EVENT_TYPE2_NAMES = {
    12: "Key pass", 13: "Failed through ball", 14: "Sending off", 15: "Own goal",
}
BODYPART_NAMES = {1: "Right foot", 2: "Left foot", 3: "Head"}
SHOT_OUTCOME_NAMES = {1: "On target", 2: "Off target", 3: "Blocked", 4: "Hit the bar"}
SITUATION_NAMES = {1: "Open play", 2: "Set piece", 3: "Corner", 4: "Free kick"}
ASSIST_METHOD_NAMES = {0: "None", 1: "Pass", 2: "Cross", 3: "Headed pass", 4: "Through ball"}
LOCATION_NAMES = {
    1: "Attacking half", 2: "Defensive half", 3: "Centre of the box",
    4: "Left wing", 5: "Right wing", 6: "Difficult angle and long range",
    7: "Difficult angle on the left", 8: "Difficult angle on the right",
    9: "Left side of the box", 10: "Left side of the six yard box",
    11: "Right side of the box", 12: "Right side of the six yard box",
    13: "Very close range", 14: "Penalty spot", 15: "Outside the box",
    16: "Long range", 17: "More than 35 yards", 18: "More than 40 yards",
    19: "Not recorded",
}

# =====================================================================
# THRESHOLD: which location codes count as "inside the box"
# Reasoning: a shot from inside the penalty area is a high-quality chance.
# These 7 codes are the locations geometrically inside the box (the six-yard
# box, the sides of the box, the centre, very close range, and the penalty spot).
# =====================================================================
IN_BOX_LOCATIONS = {3, 9, 10, 11, 12, 13, 14}


def map_event_codes(df):
    """Add readable *_name columns and binary is_* feature columns."""
    df = df.copy()

    # ---- text name columns ----
    df["event_type_name"] = df["event_type"].map(EVENT_TYPE_NAMES)
    df["event_type2_name"] = df["event_type2"].map(EVENT_TYPE2_NAMES)
    df["bodypart_name"] = df["bodypart"].map(BODYPART_NAMES)
    df["shot_outcome_name"] = df["shot_outcome"].map(SHOT_OUTCOME_NAMES)
    df["situation_name"] = df["situation"].map(SITUATION_NAMES)
    df["assist_method_name"] = df["assist_method"].map(ASSIST_METHOD_NAMES)
    df["location_name"] = df["location"].map(LOCATION_NAMES)

    # ---- binary feature columns (0/1 int) ----
    df["is_shot"] = (df["event_type"] == 1).astype(int)
    # is_goal already exists in the raw file (0/1) — keep as int
    df["is_goal"] = df["is_goal"].fillna(0).astype(int)
    df["is_key_pass"] = (df["event_type2"] == 12).astype(int)
    df["is_box_shot"] = df["location"].isin(IN_BOX_LOCATIONS).astype(int)
    df["is_left_foot"] = (df["bodypart"] == 2).astype(int)
    df["is_right_foot"] = (df["bodypart"] == 1).astype(int)
    df["is_header"] = (df["bodypart"] == 3).astype(int)
    df["is_on_target"] = (df["shot_outcome"] == 1).astype(int)
    df["is_yellow"] = (df["event_type"] == 4).astype(int)
    df["is_red"] = (df["event_type"].isin([5, 6])).astype(int)
    df["is_foul"] = (df["event_type"] == 3).astype(int)
    df["is_through_ball_assist"] = (df["assist_method"] == 4).astype(int)

    return df


def clean_events_data(df):
    """Clean the events table and attach the player matching key.

    Steps:
      1. translate codes + build binary columns (map_event_codes)
      2. add clean_name (normalized player name, for joining with players)
      3. keep all events; rows without a `player` get clean_name = NaN
         (they are legitimate team-level events such as some corners).
    """
    df = map_event_codes(df)
    df["clean_name"] = df["player"].apply(clean_player_name)
    return df.reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    from data_loader import load_events_data

    raw = load_events_data(root / "data/raw/events.csv")
    clean = clean_events_data(raw)

    out = root / "data/processed/clean_events.csv"
    clean.to_csv(out, index=False)

    # ---- quality checks ----
    print(f"clean_events: {clean.shape[0]:,} rows x {clean.shape[1]} cols -> {out.name}")
    print(f"is_goal sum:        {clean['is_goal'].sum():,}  (raw had 24,446)")
    print(f"is_shot sum:        {clean['is_shot'].sum():,}")
    print(f"is_key_pass sum:    {clean['is_key_pass'].sum():,}")
    print(f"is_box_shot sum:    {clean['is_box_shot'].sum():,}")

    # every event_type translated?
    untranslated = clean.loc[clean["event_type"].notna() & clean["event_type_name"].isna()]
    print(f"untranslated event_type rows: {len(untranslated)}")

    # do goals always have a player?
    goals_no_player = clean.loc[(clean["is_goal"] == 1) & clean["clean_name"].isna()]
    print(f"goals with empty player: {len(goals_no_player)}")

    # box shots should be a subset of shots
    box_not_shot = clean.loc[(clean["is_box_shot"] == 1) & (clean["is_shot"] == 0)]
    print(f"box_shots that are not shots: {len(box_not_shot)}")

    print("\nGoals by body part (sanity):")
    print(clean.loc[clean["is_goal"] == 1, "bodypart_name"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
