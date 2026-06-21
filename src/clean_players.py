"""
clean_players.py — Stage 3.
Clean the FC24 player table and add computed score columns.

Output: data/processed/clean_players.csv  (one row per player)

All THRESHOLDS / WEIGHTS are defined as named constants at the top and are
documented in docs/03_thresholds.md (the central thresholds document).
Player and club NAMES are kept in English; only a normalized matching key
(`clean_name`) is added for later joining with the events table.
"""

from pathlib import Path
import unicodedata
import re

import numpy as np
import pandas as pd

# =====================================================================
# THRESHOLDS & WEIGHTS  (see docs/03_thresholds.md for the reasoning)
# =====================================================================

# -- position_group: derived from the FIRST listed position --
# Reasoning: a player's primary position is the first token of player_positions.
POSITION_TO_GROUP = {
    "GK": "GK",
    "CB": "Defender", "RB": "Defender", "LB": "Defender",
    "RWB": "Defender", "LWB": "Defender",
    "CDM": "Midfielder", "CM": "Midfielder", "CAM": "Midfielder",
    "RM": "Midfielder", "LM": "Midfielder",
    "RW": "Forward", "LW": "Forward", "CF": "Forward", "ST": "Forward",
}

# -- ability_score weights per position group --
# Reasoning: each role is judged by the attributes that matter for that role.
# Weights sum to 1.0. Forwards reward shooting/pace; defenders reward defending/
# physic; midfielders reward passing/dribbling. GK has no face stats, so for GK
# we fall back to the curated `overall` rating.
ABILITY_WEIGHTS = {
    "Forward":    {"shooting": 0.30, "pace": 0.20, "dribbling": 0.20,
                   "passing": 0.10, "physic": 0.10, "defending": 0.10},
    "Midfielder": {"passing": 0.30, "dribbling": 0.25, "pace": 0.15,
                   "shooting": 0.10, "defending": 0.10, "physic": 0.10},
    "Defender":   {"defending": 0.40, "physic": 0.25, "pace": 0.15,
                   "passing": 0.10, "dribbling": 0.05, "shooting": 0.05},
}
FACE_STATS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

# -- market efficiency --
# Reasoning: "efficiency" = how underpriced a player is relative to ability.
# We compare each player's ABILITY percentile to their VALUE percentile.
# Score in [-100, 100]: positive => more ability than the price suggests
# (a candidate bargain). The actual bargain *flagging* is done later by the
# Isolation Forest (Stage 11); this column is the human-readable signal.
# value_eur is log-transformed first because it is extremely right-skewed
# (mean 2.8M vs median 1.0M — see docs/02_eda.md).

# -- minimum value to be eligible for the efficiency score --
# Reasoning: players with value_eur <= this are free agents / data gaps; their
# efficiency would be meaningless. They keep the score NaN.
MIN_VALUE_EUR = 10_000


# =====================================================================
# Helpers
# =====================================================================

def clean_player_name(name):
    """Normalize a name into a matching key: lowercase, strip accents/diacritics,
    collapse whitespace. Used ONLY for joining with the events table — the
    display name stays in its original English form.

    'Mladen Petrić' -> 'mladen petric'
    """
    if not isinstance(name, str):
        return None
    # decompose accents and drop the combining marks
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = ascii_str.lower().strip()
    ascii_str = re.sub(r"[^a-z0-9 ]", " ", ascii_str)   # drop punctuation
    ascii_str = re.sub(r"\s+", " ", ascii_str).strip()
    return ascii_str or None


def assign_position_group(player_positions):
    """Map the first listed FC position to one of GK/Defender/Midfielder/Forward."""
    if not isinstance(player_positions, str) or not player_positions.strip():
        return None
    primary = player_positions.split(",")[0].strip().upper()
    return POSITION_TO_GROUP.get(primary)


def calculate_ability_score(row):
    """Position-weighted blend of the 6 face stats (0-100).
    GK -> use `overall` (no face stats available)."""
    grp = row.get("position_group")
    if grp == "GK" or grp is None:
        return float(row["overall"]) if pd.notna(row.get("overall")) else np.nan
    weights = ABILITY_WEIGHTS[grp]
    total, wsum = 0.0, 0.0
    for stat, w in weights.items():
        val = row.get(stat)
        if pd.notna(val):
            total += w * float(val)
            wsum += w
    if wsum == 0:
        return np.nan
    return round(total / wsum, 2)   # renormalize if some stat was missing


def calculate_market_efficiency_score(df):
    """Vectorized: ability percentile minus value percentile, in [-100, 100].
    Higher = more underpriced relative to ability (candidate bargain)."""
    eligible = df["value_eur"].fillna(0) >= MIN_VALUE_EUR
    ability_pct = df["ability_score"].rank(pct=True) * 100
    log_value = np.log10(df["value_eur"].where(eligible))
    value_pct = log_value.rank(pct=True) * 100
    score = (ability_pct - value_pct).round(2)
    score[~eligible] = np.nan
    return score


# =====================================================================
# Main cleaning pipeline
# =====================================================================

def clean_players_data(df):
    """Clean the FC24 table and add computed columns.

    Steps:
      1. drop duplicate player_id
      2. position_group (from primary position)
      3. clean_name (matching key for events)
      4. ability_score (position-weighted)
      5. potential_growth = potential - overall
      6. market_efficiency_score
    """
    df = df.drop_duplicates(subset="player_id", keep="first").copy()

    # position group
    df["position_group"] = df["player_positions"].apply(assign_position_group)

    # matching key (display name stays English in long_name/short_name)
    df["clean_name"] = df["long_name"].apply(clean_player_name)

    # scores
    df["ability_score"] = df.apply(calculate_ability_score, axis=1)
    df["potential_growth"] = (df["potential"] - df["overall"]).astype("Float64")
    df["market_efficiency_score"] = calculate_market_efficiency_score(df)

    return df.reset_index(drop=True)


def main():
    root = Path(__file__).resolve().parent.parent
    import sys
    sys.path.insert(0, str(root / "src"))
    from data_loader import load_players_data

    raw = load_players_data(root / "data/raw/male_players.csv")  # FC24 only
    clean = clean_players_data(raw)

    out = root / "data/processed/clean_players.csv"
    clean.to_csv(out, index=False)

    # ---- quality checks ----
    print(f"clean_players: {clean.shape[0]:,} rows x {clean.shape[1]} cols -> {out.name}")
    print("duplicate player_id:", clean["player_id"].duplicated().sum())
    print("NaN in overall:", clean["overall"].isna().sum())
    print("NaN in value_eur:", clean["value_eur"].isna().sum())
    print("\nposition_group counts:")
    print(clean["position_group"].value_counts(dropna=False).to_string())
    print("\nability_score by group (mean):")
    print(clean.groupby("position_group")["ability_score"].mean().round(2).to_string())
    print("\nTop 5 by market_efficiency_score (candidate bargains):")
    cols = ["short_name", "position_group", "overall", "ability_score",
            "value_eur", "market_efficiency_score"]
    print(clean.nlargest(5, "market_efficiency_score")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
