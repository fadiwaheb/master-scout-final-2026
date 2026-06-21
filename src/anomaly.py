"""
anomaly.py — Stage 11.  USE #2 (anomalies).
Detect (a) BARGAIN players (high ability vs low value) and (b) PROFILE-vs-
PERFORMANCE anomalies (FC24 rating disagrees with actual event output).

Algorithms (scikit-learn, ready-made — we USE, not train):
    run_isolation_forest(X, contamination)
    run_dbscan(X, eps, min_samples)
    run_one_class_svm(X, nu)              (optional 3rd, for comparison)

Key deliverable for rubric ch. 5+6: a MODEL COMPARISON — Isolation Forest vs
DBSCAN on the SAME features, documenting which outliers each method caught.

THRESHOLDS (decisions doc):
    ANOMALY_CONTAMINATION — expected fraction of anomalies (IF + OCSVM nu)
    DBSCAN_EPS / DBSCAN_MIN_SAMPLES — neighborhood for DBSCAN (on scaled features)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent))

# expected fraction of anomalies. Reasoning: bargains/mismatches are RARE; 2%
# keeps the flagged set small and high-signal (a scout reviews a short list).
ANOMALY_CONTAMINATION = 0.02

# DBSCAN neighborhood on StandardScaler'd features. Reasoning: the player data is
# a CONTINUOUS blob with no density gaps, so DBSCAN merges almost everything into
# one cluster — even a tight eps=0.6 leaves only a handful as noise. This is
# itself a finding (DBSCAN is ill-suited to continuous data; IF is the right tool
# here) and is documented in the model comparison.
DBSCAN_EPS = 0.6
DBSCAN_MIN_SAMPLES = 6

# feature sets
BARGAIN_FEATURES = ["overall", "potential", "age", "log_value"]
MISMATCH_FEATURES = ["overall", "attacking_involvement_score",
                     "goals_per_match", "conversion_rate"]


# ---------------------------------------------------------------------
# generic algorithm wrappers (return: anomaly flag array, score where available)
# ---------------------------------------------------------------------
def run_isolation_forest(X, contamination=ANOMALY_CONTAMINATION):
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    labels = model.fit_predict(X)               # -1 anomaly, 1 normal
    scores = model.decision_function(X)         # lower = more anomalous
    return (labels == -1), scores


def run_dbscan(X, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return (labels == -1), labels               # -1 = noise = anomaly


def run_one_class_svm(X, nu=ANOMALY_CONTAMINATION):
    labels = OneClassSVM(nu=nu, kernel="rbf", gamma="scale").fit_predict(X)
    return (labels == -1), None


def _scaled(df, feats):
    """Drop rows with missing features, return (pool, scaled X)."""
    pool = df.dropna(subset=feats).copy()
    X = StandardScaler().fit_transform(pool[feats].astype(float))
    return pool, X


# ---------------------------------------------------------------------
# Use 2a — bargain players (high ability vs low value)
# ---------------------------------------------------------------------
def detect_bargain_players(df, contamination=ANOMALY_CONTAMINATION,
                           min_overall=None, max_age=None, max_value_eur=None,
                           top_n=20):
    """Isolation Forest over [overall, potential, age, log_value] (outfield only);
    among the statistical anomalies, keep the UNDERPRICED ones
    (market_efficiency_score>0) and rank them — lots of ability for the money.

    Optional min_overall / max_age narrow the list to realistic targets (e.g.
    min_overall=78 surfaces elite-but-cheap veterans like Chiellini, Ramos)."""
    work = df[(df["value_eur"].notna()) & (df["value_eur"] > 0)
              & (df["position_group"] != "GK")].copy()
    work["log_value"] = np.log10(work["value_eur"])
    pool, X = _scaled(work, BARGAIN_FEATURES)

    is_anom, scores = run_isolation_forest(X, contamination)
    pool = pool.assign(is_anomaly=is_anom, anomaly_score=np.round(scores, 4))

    bargains = pool[pool["is_anomaly"] & (pool["market_efficiency_score"] > 0)]
    if min_overall is not None:
        bargains = bargains[bargains["overall"] >= min_overall]
    if max_age is not None:
        bargains = bargains[bargains["age"] <= max_age]
    if max_value_eur is not None:
        bargains = bargains[bargains["value_eur"] <= max_value_eur]

    cols = ["short_name", "position_group", "age", "overall", "potential",
            "value_eur", "market_efficiency_score", "anomaly_score"]
    return bargains.sort_values("market_efficiency_score", ascending=False
                                ).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Use 2b — profile vs performance mismatch (needs event data)
# ---------------------------------------------------------------------
def detect_profile_performance_anomalies(df, contamination=0.05, top_n=20):
    """Isolation Forest over [overall, attacking_involvement_score,
    goals_per_match, conversion_rate] for players WITH event data. Flags players
    whose FC24 rating disagrees with their real output (over/under-performers)."""
    work = df[df["has_event_data"] & (df["position_group"] != "GK")].copy()
    pool, X = _scaled(work, MISMATCH_FEATURES)

    is_anom, scores = run_isolation_forest(X, contamination)
    pool = pool.assign(is_anomaly=is_anom, anomaly_score=np.round(scores, 4))
    anomalies = pool[pool["is_anomaly"]].copy()

    # label the mismatch direction
    s = anomalies
    conds = [
        (s["attacking_involvement_score"] >= 65) & (s["overall"] <= 74)
            & (s["goals_per_match"] > 0.2),
        (s["overall"] >= 85) & (s["attacking_involvement_score"] >= 90),
        (s["overall"] >= 80) & (s["attacking_involvement_score"] < 50),
    ]
    choices = [
        "over-performer (hidden gem: output > rating)",
        "elite (extreme but consistent output)",
        "under-performer (rating > output)",
    ]
    anomalies["direction"] = np.select(conds, choices, default="unusual profile")

    cols = ["short_name", "position_group", "age", "overall",
            "attacking_involvement_score", "goals_per_match", "conversion_rate",
            "direction", "anomaly_score"]
    return anomalies.sort_values("anomaly_score")[cols].head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------
# Model comparison — Isolation Forest vs DBSCAN (vs One-Class SVM)
# ---------------------------------------------------------------------
def compare_if_vs_dbscan(df, contamination=ANOMALY_CONTAMINATION):
    """Run IF, DBSCAN and One-Class SVM on the SAME bargain feature space and
    report how many anomalies each found and how much they overlap."""
    work = df[(df["value_eur"].notna()) & (df["value_eur"] > 0)].copy()
    work["log_value"] = np.log10(work["value_eur"])
    pool, X = _scaled(work, BARGAIN_FEATURES)

    if_anom, _ = run_isolation_forest(X, contamination)
    db_anom, _ = run_dbscan(X)
    ocsvm_anom, _ = run_one_class_svm(X, nu=contamination)

    s_if = set(pool.index[if_anom])
    s_db = set(pool.index[db_anom])
    s_oc = set(pool.index[ocsvm_anom])
    return {
        "isolation_forest": len(s_if),
        "dbscan": len(s_db),
        "one_class_svm": len(s_oc),
        "IF∩DBSCAN": len(s_if & s_db),
        "IF∩OCSVM": len(s_if & s_oc),
        "all_three": len(s_if & s_db & s_oc),
        "total_players": len(pool),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    print("=" * 70, "\nSTAGE 11 — Anomaly detection (USE #2)\n", "=" * 70)

    print("\n[2a] BARGAIN players, elite-but-cheap (min_overall=78):")
    b = detect_bargain_players(df, min_overall=78, top_n=8)
    print(b.to_string(index=False))
    print("  quality: all market_efficiency_score > 0:",
          bool((b["market_efficiency_score"] > 0).all()))

    print("\n[2b] PROFILE-vs-PERFORMANCE anomalies (players with event data):")
    a = detect_profile_performance_anomalies(df, top_n=8)
    print(a.to_string(index=False))

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Isolation Forest vs DBSCAN vs One-Class SVM")
    print("=" * 70)
    cmp = compare_if_vs_dbscan(df)
    for k, v in cmp.items():
        print(f"  {k:<16}: {v:,}")
    print("Interpretation: IF flags a fixed ~2% by isolation depth; DBSCAN flags "
          "density-based noise (count varies with eps); their intersection are the "
          "most robust outliers. One-Class SVM gives a third boundary for comparison.")


if __name__ == "__main__":
    main()
