"""
player_event_stats.py — Stage 6.
Aggregate player_match_stats to ONE row per player, with totals, per-match
values, rates (0-1), four computed scores, and matches_with_2_plus_goals.

Output: data/processed/player_event_stats.csv  (one row per player)

THRESHOLDS / WEIGHTS (documented in docs/ספים_והחלטות_Master_Scout.docx):
  MIN_MATCHES        — minimum matches for reliable rates/scores
  MIN_FOOT_SHOTS     — minimum foot shots for a meaningful foot-balance score
  DISCIPLINE_CARD_CAP— weighted cards/match that maps to discipline_score = 0
  ATTACKING_WEIGHTS / CREATIVE_WEIGHTS — weights inside the raw attacking/creative blends
"""

from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================================
# THRESHOLDS & WEIGHTS
# =====================================================================

# Minimum matches to trust a player's rates/scores.
# Reasoning: rates from 1-2 matches are noise (a 1-match player with 1 goal
# would show goals_per_match=1.0). Below this, the percentile scores are NaN.
MIN_MATCHES = 5

# Minimum foot shots (left+right) for a foot-balance score.
# Reasoning: you cannot judge two-footedness from 1-2 shots.
MIN_FOOT_SHOTS = 5

# Weighted cards per match that maps discipline_score to 0 (worst).
# Reasoning: a player averaging >= 0.5 weighted cards per match is highly
# undisciplined; 0 cards -> score 100. (yellow=1, red=3 weight.)
DISCIPLINE_CARD_CAP = 0.5
RED_CARD_WEIGHT = 3

# Raw attacking blend (per-match), goals weighted highest. Turned into a
# 0-100 percentile across qualified players.
ATTACKING_WEIGHTS = {
    "goals_per_match": 3.0,
    "key_passes_per_match": 2.0,
    "box_shots_per_match": 1.0,
    "shots_per_match": 0.5,
}
# Raw creative blend (per-match); through balls are rarer/more creative.
CREATIVE_WEIGHTS = {
    "through_ball_assists_per_match": 3.0,
    "key_passes_per_match": 1.0,
}

# columns in player_match_stats to sum into season totals
TOTAL_MAP = {
    "shots": "total_shots",
    "goals": "total_goals",
    "key_passes": "total_key_passes",
    "box_shots": "total_box_shots",
    "shots_on_target": "total_shots_on_target",
    "left_foot_shots": "total_left_foot_shots",
    "right_foot_shots": "total_right_foot_shots",
    "header_shots": "total_header_shots",
    "yellow_cards": "total_yellow_cards",
    "red_cards": "total_red_cards",
    "fouls": "total_fouls",
    "through_ball_assists": "total_through_ball_assists",
}


def _safe_div(a, b):
    """Element-wise a/b, returning NaN where b == 0."""
    b = b.replace(0, np.nan)
    return a / b


# ---- the four computed scores ----

def calculate_foot_balance_score(df):
    """0-100 two-footedness from shot feet. 100 = perfectly balanced.
    NaN if foot shots < MIN_FOOT_SHOTS."""
    left = df["total_left_foot_shots"]
    right = df["total_right_foot_shots"]
    foot_total = left + right
    weaker = np.minimum(left, right)
    stronger = np.maximum(left, right).replace(0, np.nan)
    score = (weaker / stronger) * 100      # 0 = one-footed, 100 = even
    score[foot_total < MIN_FOOT_SHOTS] = np.nan
    return score.round(2)


def calculate_discipline_score(df):
    """0-100 discipline. 100 = never booked; 0 = >= DISCIPLINE_CARD_CAP
    weighted cards per match."""
    weighted_cards = df["total_yellow_cards"] + RED_CARD_WEIGHT * df["total_red_cards"]
    cards_per_match = weighted_cards / df["matches"]
    score = 100 * (1 - np.minimum(cards_per_match / DISCIPLINE_CARD_CAP, 1.0))
    score[df["matches"] < MIN_MATCHES] = np.nan
    return score.round(2)


def _weighted_percentile_score(df, weights):
    """Build a raw weighted blend, then convert to a 0-100 percentile across
    players with matches >= MIN_MATCHES. Others get NaN."""
    qualified = df["matches"] >= MIN_MATCHES
    raw = sum(w * df[col] for col, w in weights.items())
    raw = raw.where(qualified)
    pct = raw.rank(pct=True) * 100
    return pct.round(2)


def calculate_attacking_involvement_score(df):
    """0-100 percentile of a per-match attacking blend (goals-heavy)."""
    return _weighted_percentile_score(df, ATTACKING_WEIGHTS)


def calculate_creative_score(df):
    """0-100 percentile of a per-match creativity blend (key passes / through balls)."""
    return _weighted_percentile_score(df, CREATIVE_WEIGHTS)


def build_player_event_stats(pms):
    """Aggregate player_match_stats (player x match) to one row per player."""
    g = pms.groupby("clean_name")

    out = g.agg(**{name: (src, "sum") for src, name in TOTAL_MAP.items()})
    out["matches"] = g["id_odsp"].nunique()
    out["matches_with_2_plus_goals"] = g["goals"].apply(lambda s: int((s >= 2).sum()))
    out = out.reset_index()

    # ---- per-match columns ----
    out["goals_per_match"] = _safe_div(out["total_goals"], out["matches"])
    out["shots_per_match"] = _safe_div(out["total_shots"], out["matches"])
    out["key_passes_per_match"] = _safe_div(out["total_key_passes"], out["matches"])
    out["box_shots_per_match"] = _safe_div(out["total_box_shots"], out["matches"])
    out["through_ball_assists_per_match"] = _safe_div(
        out["total_through_ball_assists"], out["matches"])

    # ---- rates (0-1) ----
    out["shot_accuracy"] = _safe_div(out["total_shots_on_target"], out["total_shots"])
    out["conversion_rate"] = _safe_div(out["total_goals"], out["total_shots"])
    out["box_shot_rate"] = _safe_div(out["total_box_shots"], out["total_shots"])

    # ---- four computed scores ----
    out["foot_balance_score"] = calculate_foot_balance_score(out)
    out["discipline_score"] = calculate_discipline_score(out)
    out["attacking_involvement_score"] = calculate_attacking_involvement_score(out)
    out["creative_score"] = calculate_creative_score(out)

    return out


def main():
    root = Path(__file__).resolve().parent.parent
    pms = pd.read_csv(root / "data/processed/player_match_stats.csv")
    pes = build_player_event_stats(pms)

    out = root / "data/processed/player_event_stats.csv"
    pes.to_csv(out, index=False)

    # ---- quality checks ----
    print(f"player_event_stats: {pes.shape[0]:,} rows x {pes.shape[1]} cols -> {out.name}")
    print("reconcile goals_per_match = total_goals/matches:",
          bool(np.allclose((pes["total_goals"] / pes["matches"]),
                           pes["goals_per_match"], equal_nan=True)))
    for c in ["shot_accuracy", "conversion_rate", "box_shot_rate"]:
        v = pes[c].dropna()
        print(f"  {c}: min={v.min():.3f} max={v.max():.3f}  (expect 0..1)")
    print("any rate > 1:", int(((pes[["shot_accuracy", "conversion_rate", "box_shot_rate"]] > 1)
                                 .sum().sum())))
    print("players with matches >= MIN_MATCHES:", int((pes["matches"] >= MIN_MATCHES).sum()))
    print("\nTop 5 attacking_involvement_score:")
    cols = ["clean_name", "matches", "total_goals", "goals_per_match",
            "attacking_involvement_score", "creative_score"]
    print(pes.nlargest(5, "attacking_involvement_score")[cols].to_string(index=False))
    print("\nTop 5 creative_score:")
    print(pes.nlargest(5, "creative_score")[cols].to_string(index=False))
    print("\nMost two-footed (foot_balance_score, min foot shots applied):")
    fb = pes.dropna(subset=["foot_balance_score"])
    print(fb.nlargest(5, "foot_balance_score")[
        ["clean_name", "total_left_foot_shots", "total_right_foot_shots",
         "foot_balance_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
