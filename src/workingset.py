"""
workingset.py — the "funnel" engine.

Lets the chat narrow the big table to a smaller WORKING SET, query after query.
Example: "forwards 25-30"  ->  then "of those, who has 15+ braces"  ->  then ...
Each step returns the FULL-column subset of the players that match so far, so the
next query can filter on any column. "start over" resets to the full table.

The chat passes the current working set as the source to the agent, and after a
successful query calls narrow() to compute the next working set.
"""

from __future__ import annotations

import pandas as pd

# intents that narrow by FILTERING rows (every matching player stays in the set)
_FILTER_NARROW = {
    "profile_search", "attacking_players", "creative_midfielders",
    "disciplined_defenders", "two_footed", "braces",
}
# intents that narrow to the MODEL-RANKED players that were shown
_RESULT_NARROW = {"similar_players", "bargains", "profile_performance_anomaly"}

# implicit position for the position-specific performance intents
_IMPLICIT_POS = {"creative_midfielders": "Midfielder",
                 "disciplined_defenders": "Defender"}

# curated "scout view" columns (only those present are shown) + Hebrew legend
SCOUT_COLS = [
    "short_name", "position_group", "age", "overall", "potential", "value_eur",
    "preferred_foot", "club_name", "league_name", "nationality_name",
    "matches", "total_goals", "goals_per_match", "matches_with_2_plus_goals",
    "attacking_involvement_score", "creative_score", "discipline_score",
    "foot_balance_score",
]
LEGEND = {
    "short_name": "שם השחקן", "position_group": "עמדה", "age": "גיל",
    "overall": "דירוג כללי (FC24)", "potential": "פוטנציאל", "value_eur": "שווי שוק (€)",
    "preferred_foot": "רגל חזקה", "club_name": "מועדון", "league_name": "ליגה",
    "nationality_name": "נבחרת", "matches": "משחקים (אירועים)", "total_goals": "גולים",
    "goals_per_match": "גולים למשחק", "matches_with_2_plus_goals": "משחקי דאבל (2+ גולים)",
    "attacking_involvement_score": "מעורבות התקפית", "creative_score": "יצירתיות",
    "discipline_score": "משמעת", "foot_balance_score": "איזון דו-רגלי",
}


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Apply every recognized ROW filter present in `f`; return the full subset."""
    out = df
    rng = [  # (column, filter_key, ">=" or "<=")
        ("age", "min_age", "ge"), ("age", "max_age", "le"),
        ("overall", "min_overall", "ge"), ("potential", "min_potential", "ge"),
        ("pace", "min_pace", "ge"),
        ("value_eur", "min_value_eur", "ge"), ("value_eur", "max_value_eur", "le"),
        # real match performance
        ("total_goals", "min_total_goals", "ge"),
        ("matches_with_2_plus_goals", "min_braces", "ge"),
        ("goals_per_match", "min_goals_per_match", "ge"),
        ("total_key_passes", "min_key_passes", "ge"),       # ≈ assists / chance creation
        ("total_shots", "min_shots", "ge"),
        ("total_yellow_cards", "max_yellow_cards", "le"),
        ("total_red_cards", "max_red_cards", "le"),
        ("attacking_involvement_score", "min_attacking_involvement", "ge"),
        ("creative_score", "min_creative", "ge"),
        ("discipline_score", "min_discipline", "ge"),
        ("foot_balance_score", "min_foot_balance", "ge"),
    ]
    if f.get("position_group") and "position_group" in out:
        out = out[out["position_group"] == f["position_group"]]
    if f.get("preferred_foot") and "preferred_foot" in out:
        out = out[out["preferred_foot"] == f["preferred_foot"]]
    if f.get("league_name") and "league_name" in out:
        out = out[out["league_name"].str.contains(f["league_name"], case=False, na=False)]
    if f.get("nationality") and "nationality_name" in out:
        out = out[out["nationality_name"].str.contains(f["nationality"], case=False, na=False)]
    for col, key, op in rng:
        val = f.get(key)
        if val is not None and col in out.columns:
            out = out[out[col] >= val] if op == "ge" else out[out[col] <= val]
    return out


# performance filters require event data; each also implies a sensible sort metric
_PERF_KEYS = {"min_total_goals", "min_braces", "min_goals_per_match", "min_key_passes",
              "min_shots", "max_yellow_cards", "max_red_cards",
              "min_attacking_involvement", "min_creative", "min_discipline",
              "min_foot_balance"}
_SORT_DEFAULT = {
    "profile_search": "overall", "attacking_players": "attacking_involvement_score",
    "creative_midfielders": "creative_score", "disciplined_defenders": "discipline_score",
    "two_footed": "foot_balance_score", "braces": "matches_with_2_plus_goals",
}
_FILTER_SORT = {
    "min_total_goals": "total_goals", "min_key_passes": "total_key_passes",
    "min_shots": "total_shots", "min_braces": "matches_with_2_plus_goals",
    "min_goals_per_match": "goals_per_match",
}
DISPLAY_BASE = ["short_name", "position_group", "age", "overall", "potential", "value_eur"]


def _prepare(df: pd.DataFrame, intent: str, f: dict):
    """Add implicit constraints (event data for performance intents/filters, the
    implicit position for creative/disciplined). Returns (base_df, filters)."""
    base = df
    needs_events = (intent in (_FILTER_NARROW - {"profile_search"})
                    or any(k in f for k in _PERF_KEYS))
    if needs_events and "has_event_data" in df.columns:
        base = df[df["has_event_data"]]
    if intent in _IMPLICIT_POS and "position_group" not in f:
        f = {**f, "position_group": _IMPLICIT_POS[intent]}
    return base, f


def search(df: pd.DataFrame, intent: str, filters: dict, top_n: int = 10):
    """Generic filter + sort + top_n. Returns (display_df, full_narrowed_df).
    Any recognized filter works here — this is what makes free text robust."""
    base, f = _prepare(df, intent, dict(filters or {}))
    sub = _apply_filters(base, f)
    sort_by = next((_FILTER_SORT[k] for k in _FILTER_SORT if k in f),
                   _SORT_DEFAULT.get(intent, "overall"))
    if sort_by not in sub.columns:
        sort_by = "overall"
    sub = sub.sort_values(sort_by, ascending=False)
    cols = list(DISPLAY_BASE)
    if sort_by not in cols:
        cols.append(sort_by)
    cols = [c for c in cols if c in sub.columns]
    return sub.head(top_n)[cols].reset_index(drop=True), sub


def narrow(source: pd.DataFrame, intent: str, filters: dict,
           display_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Return the next working set (full columns), or None to leave it unchanged."""
    f = dict(filters or {})
    if intent in _FILTER_NARROW:
        base, f = _prepare(source, intent, f)
        return _apply_filters(base, f)

    if (intent in _RESULT_NARROW and display_df is not None
            and "short_name" in display_df.columns and "short_name" in source.columns):
        return source[source["short_name"].isin(display_df["short_name"])]

    return None  # clustering / visualize / greeting -> working set unchanged


def scout_view(df: pd.DataFrame) -> pd.DataFrame:
    """Curated, readable column subset of a working set for display."""
    cols = [c for c in SCOUT_COLS if c in df.columns]
    return df[cols]
