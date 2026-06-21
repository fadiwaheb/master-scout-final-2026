"""
player_match_stats.py — Stage 5.
Aggregate clean_events to the player-in-a-single-match level.

Output: data/processed/player_match_stats.csv  (one row per player x match)

Grouping key: clean_name + id_odsp  (a player's line in one specific match).
This table is the basis for "braces" (matches with 2+ goals) in Stage 6.
No numeric thresholds here — pure aggregation of the binary columns from Stage 4.
"""

from pathlib import Path
import sys

import pandas as pd

# binary columns produced in Stage 4 that we sum up per player-match
SUM_COLUMNS = {
    "is_shot": "shots",
    "is_goal": "goals",
    "is_key_pass": "key_passes",
    "is_box_shot": "box_shots",
    "is_on_target": "shots_on_target",
    "is_left_foot": "left_foot_shots",
    "is_right_foot": "right_foot_shots",
    "is_header": "header_shots",
    "is_yellow": "yellow_cards",
    "is_red": "red_cards",
    "is_foul": "fouls",
    "is_through_ball_assist": "through_ball_assists",
}


def build_player_match_stats(clean_events_df):
    """Aggregate events to one row per (clean_name, id_odsp).

    Drops team-level events with no player (clean_name is NaN).
    """
    df = clean_events_df.dropna(subset=["clean_name"]).copy()

    agg = (
        df.groupby(["clean_name", "id_odsp"], as_index=False)[list(SUM_COLUMNS)]
        .sum()
        .rename(columns=SUM_COLUMNS)
    )

    # carry the team the player appeared for in that match (most frequent)
    team = (
        df.groupby(["clean_name", "id_odsp"])["event_team"]
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else None)
        .reset_index()
        .rename(columns={"event_team": "match_team"})
    )
    out = agg.merge(team, on=["clean_name", "id_odsp"], how="left")

    return out


def main():
    root = Path(__file__).resolve().parent.parent
    src = root / "data/processed/clean_events.csv"

    # only load the columns we need (the file is large)
    usecols = ["clean_name", "id_odsp", "event_team"] + list(SUM_COLUMNS)
    events = pd.read_csv(src, usecols=usecols, low_memory=False)

    pms = build_player_match_stats(events)

    out = root / "data/processed/player_match_stats.csv"
    pms.to_csv(out, index=False)

    # ---- quality checks ----
    print(f"player_match_stats: {pms.shape[0]:,} rows x {pms.shape[1]} cols -> {out.name}")
    print(f"unique players: {pms['clean_name'].nunique():,}")
    print(f"unique matches: {pms['id_odsp'].nunique():,}")
    print(f"total goals (reconcile vs clean_events 24,446): {int(pms['goals'].sum()):,}")
    print(f"max goals by a player in one match: {int(pms['goals'].max())}")
    print(f"rows where goals > shots: {(pms['goals'] > pms['shots']).sum()}  "
          "(penalties/own-goals can exceed counted shots)")
    print(f"rows where box_shots > shots: {(pms['box_shots'] > pms['shots']).sum()}  (should be 0)")
    print("\nmatches with 2+ goals (braces) — sanity, top scorers' big games:")
    braces = pms[pms["goals"] >= 2]
    print(f"  total brace performances: {len(braces):,}")
    print(braces.nlargest(5, "goals")[["clean_name", "id_odsp", "goals", "shots"]].to_string(index=False))


if __name__ == "__main__":
    main()
