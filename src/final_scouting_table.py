"""
final_scouting_table.py — Stage 7.
Merge the FC24 profile (clean_players) with the event performance metrics
(player_event_stats) into ONE central table that the whole agent runs on.

Output: data/processed/final_scouting_table.csv  (one row per player)

Join: LEFT join players <- event stats on `clean_name`. Every FC24 player is
kept; those without matching event data get has_event_data=False. This is how
we handle the YEAR GAP (FC24=2023 vs events=2012-2017): coverage is partial,
and we mark it explicitly rather than dropping anyone.
"""

from pathlib import Path
import pandas as pd

# event-derived columns to bring across (the rest stay in player_event_stats.csv)
EVENT_COLUMNS = [
    "matches", "total_goals", "total_shots", "total_key_passes",
    "total_box_shots", "total_yellow_cards", "total_red_cards",
    "matches_with_2_plus_goals",
    "goals_per_match", "shots_per_match", "key_passes_per_match",
    "shot_accuracy", "conversion_rate", "box_shot_rate",
    "attacking_involvement_score", "creative_score",
    "discipline_score", "foot_balance_score",
]


def build_final_scouting_table(players, event_stats):
    """Left-join FC24 players with event stats on clean_name; add source flags."""
    # event_stats is keyed by clean_name (unique). Keep only the columns we merge.
    ev = event_stats[["clean_name"] + EVENT_COLUMNS].copy()

    merged = players.merge(ev, on="clean_name", how="left", validate="m:1")

    # source flags
    merged["has_event_data"] = merged["matches"].notna()
    merged["data_source_note"] = merged["has_event_data"].map({
        True: "FC24 profile + Football Events performance",
        False: "FC24 profile only (no event-match found; year-gap/name-mismatch)",
    })
    return merged


def main():
    root = Path(__file__).resolve().parent.parent
    players = pd.read_csv(root / "data/processed/clean_players.csv")
    event_stats = pd.read_csv(root / "data/processed/player_event_stats.csv")

    final = build_final_scouting_table(players, event_stats)
    out = root / "data/processed/final_scouting_table.csv"
    final.to_csv(out, index=False)

    # ---- quality checks ----
    n = len(final)
    matched = int(final["has_event_data"].sum())
    print(f"final_scouting_table: {n:,} rows x {final.shape[1]} cols -> {out.name}")
    print(f"duplicate player_id: {int(final['player_id'].duplicated().sum())}  (expect 0)")
    print(f"has_event_data = True:  {matched:,}  ({matched/n*100:.1f}%)")
    print(f"has_event_data = False: {n-matched:,}  ({(n-matched)/n*100:.1f}%)")

    # name collisions: FC24 players sharing a clean_name (ambiguous merge)
    dup_names = final["clean_name"].duplicated(keep=False) & final["clean_name"].notna()
    print(f"FC24 players sharing a clean_name (ambiguous): {int(dup_names.sum())}")

    # coverage by position group
    print("\nEvent-data coverage by position_group:")
    cov = final.groupby("position_group")["has_event_data"].agg(["sum", "count"])
    cov["pct"] = (cov["sum"] / cov["count"] * 100).round(1)
    print(cov.to_string())

    # sanity: a few well-known players that SHOULD match
    print("\nSpot check — top FC24 names with event data:")
    cols = ["short_name", "clean_name", "overall", "has_event_data",
            "matches", "total_goals"]
    have = final[final["has_event_data"]].nlargest(8, "overall")[cols]
    print(have.to_string(index=False))


if __name__ == "__main__":
    main()
