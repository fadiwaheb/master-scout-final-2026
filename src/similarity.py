"""
similarity.py — Stage 9.  USE #1 (similarity).
Find similar players with feature normalization + Cosine similarity.

This is the numeric analog of the TF-IDF + Cosine technique from the course:
instead of TF-IDF term weights we use MinMax-normalized attribute vectors, then
measure the cosine of the angle between players.

Key deliverable for rubric ch. 5+6: a MODEL COMPARISON — run the same query
WITH the position_group category (restrict candidates to the same position) and
WITHOUT it, and document how the results differ.

Output: similarity.py (module). Similarity score is in [0, 1] because MinMax
features are non-negative.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Attribute features used for play-style similarity (FC24, available for all
# outfield players). Chosen to span shooting, passing, dribbling, defending,
# physical and movement — a rounded play-style fingerprint.
SIM_FEATURES = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "skill_ball_control", "skill_dribbling", "mentality_vision",
    "attacking_finishing", "power_strength", "movement_sprint_speed",
    "movement_acceleration", "defending_standing_tackle",
]


# the 6 "face" attributes — the only play-style features a goalkeeper lacks and
# the only ones we can get from an external source for a player outside our data.
FACE6 = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]


def find_player_matches(df, player_name, limit=8):
    """All distinct players whose name matches (for disambiguation)."""
    q = player_name.strip().lower()
    sn, ln = df["short_name"].str.lower(), df["long_name"].str.lower()
    mask = sn.str.contains(re.escape(q), na=False) | ln.str.contains(re.escape(q), na=False)
    hits = df[mask]
    if hits.empty:  # token-AND fallback
        toks = [t for t in re.split(r"\s+", q) if t]
        if toks:
            m = pd.Series(True, index=df.index)
            for t in toks:
                m &= (sn.str.contains(re.escape(t), na=False) | ln.str.contains(re.escape(t), na=False))
            hits = df[m]
    return hits.sort_values("overall", ascending=False).head(limit)


def find_similar_to_attrs(df, attrs, top_n=10, position_group=None):
    """Cosine similarity against the pool using ONLY the 6 face attributes — used
    for a target player that is NOT in our data (attributes come from an external
    source). Returns a DataFrame with a `similarity` column."""
    pool = df[df["position_group"] != "GK"].dropna(subset=FACE6).copy()
    if position_group:
        pool = pool[pool["position_group"] == position_group]
    target = np.array([[float(attrs[a]) for a in FACE6]])
    X = np.vstack([pool[FACE6].astype(float).values, target])
    Z = StandardScaler().fit_transform(X)
    cos = cosine_similarity(Z[-1:], Z[:-1])[0]
    pool = pool.assign(similarity=np.round((cos + 1) / 2, 4))
    pool = pool.sort_values("similarity", ascending=False).head(top_n)
    cols = ["short_name", "position_group", "age", "overall", "value_eur", "similarity"]
    return pool[cols].reset_index(drop=True)


def _find_player_row(df, player_name):
    """Locate a player by a free-text name (matches short_name or long_name,
    case-insensitive). Works for short names, full names and partials —
    e.g. "Messi", "Lionel Messi" and "L. Messi" all resolve to the same row.
    If several match, return the highest-overall one."""
    q = player_name.strip().lower()
    sn = df["short_name"].str.lower()
    ln = df["long_name"].str.lower()

    # 1) direct substring match (handles "messi", "l. messi")
    mask = sn.str.contains(re.escape(q), na=False) | ln.str.contains(re.escape(q), na=False)
    hits = df[mask]

    # 2) fall back to token-AND: every word in the query appears in the name
    #    (handles "lionel messi" vs long_name "Lionel Andrés Messi Cuccittini")
    if hits.empty:
        tokens = [t for t in re.split(r"\s+", q) if t]
        if tokens:
            tok_mask = pd.Series(True, index=df.index)
            for t in tokens:
                tok_mask &= (sn.str.contains(re.escape(t), na=False)
                             | ln.str.contains(re.escape(t), na=False))
            hits = df[tok_mask]

    if hits.empty:
        return None
    return hits.sort_values("overall", ascending=False).iloc[0]


def find_similar_players(df, player_name, top_n=10, same_position=True,
                         max_age=None, max_value_eur=None):
    """Return the top_n players most similar to `player_name`.

    same_position=True  -> WITH category: candidates limited to the target's
                           position_group (the course's "with category" run).
    same_position=False -> WITHOUT category: candidates are all outfield players.

    Returns a DataFrame with a `similarity` column in [0, 1] and a short
    `reason` naming the closest shared attributes. The target is never returned.
    """
    target = _find_player_row(df, player_name)
    if target is None:
        raise ValueError(f"player not found: {player_name!r}")

    # candidate pool: outfield players with complete features
    pool = df[df["position_group"] != "GK"].dropna(subset=SIM_FEATURES).copy()
    if same_position:
        pool = pool[pool["position_group"] == target["position_group"]]
    if max_age is not None:
        pool = pool[pool["age"] <= max_age]
    if max_value_eur is not None:
        pool = pool[pool["value_eur"] <= max_value_eur]

    # make sure the target is part of the scaling space
    if target["player_id"] not in set(pool["player_id"]):
        pool = pd.concat([pool, target.to_frame().T], ignore_index=True)

    # z-score normalize each attribute (StandardScaler) so cosine compares the
    # ABOVE/BELOW-average play-style pattern — far more discriminative than raw
    # positive vectors. Cosine in [-1,1] is then rescaled to [0,1] via (x+1)/2,
    # so 1 = identical style, 0.5 = unrelated, 0 = opposite style.
    scaler = StandardScaler()
    X = scaler.fit_transform(pool[SIM_FEATURES].astype(float))

    target_idx = pool.index[pool["player_id"] == target["player_id"]][0]
    target_vec = X[pool.index.get_loc(target_idx)].reshape(1, -1)
    cos = cosine_similarity(target_vec, X)[0]
    sims = (cos + 1) / 2  # map [-1,1] -> [0,1]

    pool = pool.assign(similarity=np.round(sims, 4))
    pool = pool[pool["player_id"] != target["player_id"]]  # drop self
    pool = pool.sort_values("similarity", ascending=False).head(top_n)

    # human-readable reason: the attributes where the two players are closest
    pool["reason"] = pool.apply(
        lambda r: _reason(target, r), axis=1)

    cols = ["short_name", "position_group", "age", "overall", "value_eur",
            "similarity", "reason"]
    return pool[cols].reset_index(drop=True), target


def _reason(target, cand, k=3):
    """Name the k attributes where candidate is closest to the target."""
    diffs = {f: abs(float(target[f]) - float(cand[f])) for f in
             ["pace", "shooting", "passing", "dribbling", "defending", "physic"]}
    closest = sorted(diffs, key=diffs.get)[:k]
    return "similar " + ", ".join(closest)


def compare_with_without_category(df, player_name, top_n=10):
    """MODEL COMPARISON for the rubric: same query, with vs without the
    position_group category. Returns (with_cat, without_cat, overlap_count)."""
    with_cat, _ = find_similar_players(df, player_name, top_n=top_n, same_position=True)
    without_cat, _ = find_similar_players(df, player_name, top_n=top_n, same_position=False)
    overlap = set(with_cat["short_name"]) & set(without_cat["short_name"])
    return with_cat, without_cat, len(overlap)


def main():
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    print("=" * 70, "\nDEMO — Stage 9 Cosine similarity\n", "=" * 70)
    name = "De Bruyne"

    res, target = find_similar_players(df, name, top_n=8, same_position=True)
    print(f"\nTarget: {target['short_name']} ({target['position_group']}, "
          f"overall {target['overall']})")
    print("\nQuality checks:")
    print("  similarity range:", res['similarity'].min(), "..", res['similarity'].max(),
          "(expect 0..1)")
    print("  target returned as own match:",
          target['short_name'] in set(res['short_name']), "(expect False)")
    print("\n[WITH category] most similar to", target['short_name'], ":")
    print(res.to_string(index=False))

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — with vs without position_group category")
    print("=" * 70)
    wc, woc, overlap = compare_with_without_category(df, name, top_n=8)
    print("\n[WITHOUT category] (candidates = all outfield positions):")
    print(woc[["short_name", "position_group", "overall", "similarity", "reason"]].to_string(index=False))
    print(f"\nOverlap between the two top-8 lists: {overlap}/8")
    print("Interpretation: WITHOUT the category, players from OTHER positions with "
          "a similar statistical profile can appear; WITH the category we stay "
          "within the same role — usually more actionable for scouting.")


if __name__ == "__main__":
    main()
