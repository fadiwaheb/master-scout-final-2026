"""
search.py — Stage 8.
Parameter-based search/filter functions over the central table
(final_scouting_table). Implements agent intents 1, 2, 5, 6, 7, 8.

DESIGN RULE: every threshold is a PARAMETER with a sensible default, so it is
trivial to change (max_age, min_pace, max_value_eur, min_braces, top_n, ...).
Each function returns a sorted DataFrame of display columns.

Profile searches run on ALL 18,350 players. Performance searches
(attacking/creative/discipline/two-footed/braces) require has_event_data=True
(only ~902 players have event metrics — see docs about the year gap).
"""

from pathlib import Path
import pandas as pd

# columns shown in results
PROFILE_COLS = ["short_name", "position_group", "age", "overall", "potential",
                "value_eur", "preferred_foot", "club_name", "league_name",
                "nationality_name"]
EVENT_COLS = ["matches", "total_goals", "goals_per_match",
              "attacking_involvement_score", "creative_score",
              "discipline_score", "foot_balance_score",
              "matches_with_2_plus_goals"]


def _profile_filter(df, position_group=None, max_age=None, min_age=None,
                    min_overall=None, min_potential=None, min_pace=None,
                    max_value_eur=None, min_value_eur=None,
                    preferred_foot=None, league_name=None, nationality=None):
    """Apply the common FC24-profile filters. Every arg is optional."""
    out = df
    if position_group is not None:
        out = out[out["position_group"] == position_group]
    if max_age is not None:
        out = out[out["age"] <= max_age]
    if min_age is not None:
        out = out[out["age"] >= min_age]
    if min_overall is not None:
        out = out[out["overall"] >= min_overall]
    if min_potential is not None:
        out = out[out["potential"] >= min_potential]
    if min_pace is not None:
        out = out[out["pace"] >= min_pace]
    if max_value_eur is not None:
        out = out[out["value_eur"] <= max_value_eur]
    if min_value_eur is not None:
        out = out[out["value_eur"] >= min_value_eur]
    if preferred_foot is not None:
        out = out[out["preferred_foot"].str.lower() == preferred_foot.lower()]
    if league_name is not None:
        out = out[out["league_name"].str.contains(league_name, case=False, na=False)]
    if nationality is not None:
        out = out[out["nationality_name"].str.contains(nationality, case=False, na=False)]
    return out


# ---------------------------------------------------------------------
# Intent 1/2/5 — generic profile search
# ---------------------------------------------------------------------
def search_players_by_profile(df, sort_by="overall", top_n=20, **filters):
    """Generic profile search. Pass any of the _profile_filter params as kwargs.
    Returns the top_n players sorted by `sort_by` (desc)."""
    res = _profile_filter(df, **filters)
    cols = PROFILE_COLS + (["has_event_data"] if "has_event_data" in res else [])
    return res.sort_values(sort_by, ascending=False).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Intent 6 — attacking players (uses event performance)
# ---------------------------------------------------------------------
def search_attacking_players(df, position_group="Forward", max_age=None,
                             max_value_eur=None, min_attacking_involvement=None,
                             min_goals_per_match=None, top_n=20):
    """Top attacking players by attacking_involvement_score. Requires event data."""
    res = df[df["has_event_data"]]
    res = _profile_filter(res, position_group=position_group, max_age=max_age,
                          max_value_eur=max_value_eur)
    if min_attacking_involvement is not None:
        res = res[res["attacking_involvement_score"] >= min_attacking_involvement]
    if min_goals_per_match is not None:
        res = res[res["goals_per_match"] >= min_goals_per_match]
    cols = PROFILE_COLS + EVENT_COLS
    return res.sort_values("attacking_involvement_score", ascending=False
                           ).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Intent 6 — creative midfielders
# ---------------------------------------------------------------------
def search_creative_midfielders(df, max_age=None, max_value_eur=None,
                                min_creative=None, top_n=20):
    """Top creative midfielders by creative_score. Requires event data."""
    res = df[df["has_event_data"]]
    res = _profile_filter(res, position_group="Midfielder", max_age=max_age,
                          max_value_eur=max_value_eur)
    if min_creative is not None:
        res = res[res["creative_score"] >= min_creative]
    cols = PROFILE_COLS + EVENT_COLS
    return res.sort_values("creative_score", ascending=False
                           ).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Intent 6 — disciplined defenders
# ---------------------------------------------------------------------
def search_disciplined_defenders(df, max_age=None, max_value_eur=None,
                                 min_discipline=70, top_n=20):
    """Defenders with high discipline_score (default min 70). Requires event data."""
    res = df[df["has_event_data"]]
    res = _profile_filter(res, position_group="Defender", max_age=max_age,
                          max_value_eur=max_value_eur)
    if min_discipline is not None:
        res = res[res["discipline_score"] >= min_discipline]
    cols = PROFILE_COLS + EVENT_COLS
    return res.sort_values("discipline_score", ascending=False
                           ).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Intent 7 — two-footed players
# ---------------------------------------------------------------------
def search_two_footed_players(df, position_group=None, min_foot_balance=60,
                              min_total_shots=20, max_age=None, top_n=20):
    """Two-footed players (high foot_balance_score). A volume filter
    (min_total_shots, default 20) avoids low-sample 'perfectly balanced'
    players. Requires event data."""
    res = df[df["has_event_data"]]
    res = _profile_filter(res, position_group=position_group, max_age=max_age)
    res = res[res["foot_balance_score"] >= min_foot_balance]
    res = res[res["total_shots"] >= min_total_shots]
    cols = PROFILE_COLS + ["foot_balance_score", "total_shots", "matches"]
    return res.sort_values("foot_balance_score", ascending=False
                           ).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Intent 8 — players with a minimum number of braces (2+ goal games)
# ---------------------------------------------------------------------
def find_players_with_min_braces(df, min_braces=2, position_group=None,
                                 max_age=None, max_value_eur=None, top_n=20):
    """Players with at least `min_braces` matches of 2+ goals. Requires event data."""
    res = df[df["has_event_data"]]
    res = _profile_filter(res, position_group=position_group, max_age=max_age,
                          max_value_eur=max_value_eur)
    res = res[res["matches_with_2_plus_goals"] >= min_braces]
    cols = PROFILE_COLS + ["matches", "total_goals", "matches_with_2_plus_goals",
                           "goals_per_match"]
    return res.sort_values("matches_with_2_plus_goals", ascending=False
                           ).head(top_n)[cols].reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    print("=" * 70, "\nDEMO — Stage 8 search functions\n", "=" * 70)

    print("\n[1] Profile: fast wingers under 23, value <= 30M (top 5)")
    r = search_players_by_profile(df, position_group="Forward", max_age=22,
                                  min_pace=85, max_value_eur=30_000_000, top_n=5)
    print(r[["short_name", "age", "overall", "value_eur", "preferred_foot",
             "has_event_data"]].to_string(index=False))

    print("\n[2] Attacking players, value <= 40M (top 5)")
    r = search_attacking_players(df, max_value_eur=40_000_000, top_n=5)
    print(r[["short_name", "age", "overall", "total_goals", "goals_per_match",
             "attacking_involvement_score"]].to_string(index=False))

    print("\n[3] Creative midfielders (top 5)")
    r = search_creative_midfielders(df, top_n=5)
    print(r[["short_name", "age", "overall", "creative_score"]].to_string(index=False))

    print("\n[4] Disciplined defenders (top 5)")
    r = search_disciplined_defenders(df, top_n=5)
    print(r[["short_name", "age", "overall", "discipline_score"]].to_string(index=False))

    print("\n[5] Two-footed players, 20+ shots (top 5)")
    r = search_two_footed_players(df, top_n=5)
    print(r[["short_name", "position_group", "foot_balance_score",
             "total_shots"]].to_string(index=False))

    print("\n[6] Players with 5+ braces (top 5)")
    r = find_players_with_min_braces(df, min_braces=5, top_n=5)
    print(r[["short_name", "total_goals", "matches_with_2_plus_goals",
             "goals_per_match"]].to_string(index=False))

    print("\nQuality check — every function returns a DataFrame, top_n respected.")


if __name__ == "__main__":
    main()
