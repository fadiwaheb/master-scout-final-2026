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

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא sys כדי להוסיף את src לנתיב
import sys

# ייבוא numpy לחישובים מספריים ולוג
import numpy as np
# ייבוא pandas לעבודה עם הטבלה
import pandas as pd
# ייבוא StandardScaler לנרמול z-score
from sklearn.preprocessing import StandardScaler
# ייבוא Isolation Forest לזיהוי חריגות לפי בידוד
from sklearn.ensemble import IsolationForest
# ייבוא DBSCAN לזיהוי חריגות לפי צפיפות
from sklearn.cluster import DBSCAN
# ייבוא One-Class SVM כשיטת השוואה שלישית
from sklearn.svm import OneClassSVM

# מוסיפים את תיקיית הקובץ לנתיב הייבוא
sys.path.insert(0, str(Path(__file__).resolve().parent))

# expected fraction of anomalies. Reasoning: bargains/mismatches are RARE; 2%
# keeps the flagged set small and high-signal (a scout reviews a short list).
# שיעור החריגות הצפוי (2%) — שומר על רשימה קצרה ובעלת אות חזק
ANOMALY_CONTAMINATION = 0.02

# DBSCAN neighborhood on StandardScaler'd features. Reasoning: the player data is
# a CONTINUOUS blob with no density gaps, so DBSCAN merges almost everything into
# one cluster — even a tight eps=0.6 leaves only a handful as noise. This is
# itself a finding (DBSCAN is ill-suited to continuous data; IF is the right tool
# here) and is documented in the model comparison.
# רדיוס השכנות של DBSCAN (על תכונות מנורמלות)
DBSCAN_EPS = 0.6
# מספר השכנים המינימלי להגדרת ליבה ב-DBSCAN
DBSCAN_MIN_SAMPLES = 6

# feature sets
# תכונות לזיהוי מציאות: דירוג, פוטנציאל, גיל ושווי בלוג
BARGAIN_FEATURES = ["overall", "potential", "age", "log_value"]
# תכונות לזיהוי אי-התאמה פרופיל-ביצועים
MISMATCH_FEATURES = ["overall", "attacking_involvement_score",
                     "goals_per_match", "conversion_rate"]


# ---------------------------------------------------------------------
# generic algorithm wrappers (return: anomaly flag array, score where available)
# ---------------------------------------------------------------------
# עוטף Isolation Forest: מחזיר מסכת חריגות וציון (נמוך = חריג יותר)
def run_isolation_forest(X, contamination=ANOMALY_CONTAMINATION):
    # בונים את המודל עם שיעור החריגות הצפוי וזרע קבוע
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    # מתאימים ומנבאים: -1 חריג, 1 תקין
    labels = model.fit_predict(X)               # -1 anomaly, 1 normal
    # ציון ההחלטה: ככל שנמוך יותר — חריג יותר
    scores = model.decision_function(X)         # lower = more anomalous
    # מחזירים מסכת חריגות וציונים
    return (labels == -1), scores


# עוטף DBSCAN: מחזיר מסכת רעש (חריגות) ואת תוויות הקלאסטרים
def run_dbscan(X, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    # מריצים DBSCAN ומקבלים תוויות (-1 = רעש)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    # מחזירים מסכת רעש ואת התוויות
    return (labels == -1), labels               # -1 = noise = anomaly


# עוטף One-Class SVM: מחזיר מסכת חריגות (אין ציון)
def run_one_class_svm(X, nu=ANOMALY_CONTAMINATION):
    # מתאימים ומנבאים: -1 חריג
    labels = OneClassSVM(nu=nu, kernel="rbf", gamma="scale").fit_predict(X)
    # מחזירים מסכת חריגות (ללא ציון)
    return (labels == -1), None


# פונקציית עזר: מסירה שורות עם תכונות חסרות ומחזירה (מאגר, מטריצה מנורמלת)
def _scaled(df, feats):
    """Drop rows with missing features, return (pool, scaled X)."""
    # מסירים שורות שחסרות בהן התכונות הנדרשות
    pool = df.dropna(subset=feats).copy()
    # מנרמלים z-score את התכונות
    X = StandardScaler().fit_transform(pool[feats].astype(float))
    # מחזירים את המאגר והמטריצה
    return pool, X


# ---------------------------------------------------------------------
# Use 2a — bargain players (high ability vs low value)
# ---------------------------------------------------------------------
# מזהה שחקני מציאות: חריגים סטטיסטיים שהם גם תת-מתומחרים
def detect_bargain_players(df, contamination=ANOMALY_CONTAMINATION,
                           min_overall=None, max_age=None, max_value_eur=None,
                           top_n=20):
    """Isolation Forest over [overall, potential, age, log_value] (outfield only);
    among the statistical anomalies, keep the UNDERPRICED ones
    (market_efficiency_score>0) and rank them — lots of ability for the money.

    Optional min_overall / max_age narrow the list to realistic targets (e.g.
    min_overall=78 surfaces elite-but-cheap veterans like Chiellini, Ramos)."""
    # עובדים רק על שחקני שדה עם שווי חיובי
    work = df[(df["value_eur"].notna()) & (df["value_eur"] > 0)
              & (df["position_group"] != "GK")].copy()
    # מוסיפים עמודת שווי בלוג (כי השווי מוטה ימינה)
    work["log_value"] = np.log10(work["value_eur"])
    # מכינים מאגר ומטריצה מנורמלת לפי תכונות המציאות
    pool, X = _scaled(work, BARGAIN_FEATURES)

    # מריצים Isolation Forest
    is_anom, scores = run_isolation_forest(X, contamination)
    # מוסיפים עמודות דגל חריגות וציון
    pool = pool.assign(is_anomaly=is_anom, anomaly_score=np.round(scores, 4))

    # מהחריגים שומרים רק את התת-מתומחרים (ציון יעילות שוק חיובי)
    bargains = pool[pool["is_anomaly"] & (pool["market_efficiency_score"] > 0)]
    # מסנן דירוג מינימלי אופציונלי
    if min_overall is not None:
        bargains = bargains[bargains["overall"] >= min_overall]
    # מסנן גיל מקסימלי אופציונלי
    if max_age is not None:
        bargains = bargains[bargains["age"] <= max_age]
    # מסנן שווי מקסימלי אופציונלי
    if max_value_eur is not None:
        bargains = bargains[bargains["value_eur"] <= max_value_eur]

    # עמודות התצוגה
    cols = ["short_name", "position_group", "age", "overall", "potential",
            "value_eur", "market_efficiency_score", "anomaly_score"]
    # ממיינים לפי יעילות שוק ומחזירים top_n
    return bargains.sort_values("market_efficiency_score", ascending=False
                                ).head(top_n)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Use 2b — profile vs performance mismatch (needs event data)
# ---------------------------------------------------------------------
# מזהה אי-התאמה בין דירוג FC24 לביצועים אמיתיים (דורש נתוני אירועים)
def detect_profile_performance_anomalies(df, contamination=0.05, top_n=20):
    """Isolation Forest over [overall, attacking_involvement_score,
    goals_per_match, conversion_rate] for players WITH event data. Flags players
    whose FC24 rating disagrees with their real output (over/under-performers)."""
    # עובדים על שחקני שדה עם נתוני אירועים
    work = df[df["has_event_data"] & (df["position_group"] != "GK")].copy()
    # מכינים מאגר ומטריצה לפי תכונות האי-התאמה
    pool, X = _scaled(work, MISMATCH_FEATURES)

    # מריצים Isolation Forest
    is_anom, scores = run_isolation_forest(X, contamination)
    # מוסיפים עמודות דגל וציון
    pool = pool.assign(is_anomaly=is_anom, anomaly_score=np.round(scores, 4))
    # שומרים רק את החריגים
    anomalies = pool[pool["is_anomaly"]].copy()

    # label the mismatch direction
    # מתייגים את כיוון אי-ההתאמה
    s = anomalies
    # תנאים לכל סוג אי-התאמה
    conds = [
        # מבצע-יתר: תפוקה גבוהה אך דירוג נמוך
        (s["attacking_involvement_score"] >= 65) & (s["overall"] <= 74)
            & (s["goals_per_match"] > 0.2),
        # עילית: דירוג גבוה ותפוקה גבוהה מאוד
        (s["overall"] >= 85) & (s["attacking_involvement_score"] >= 90),
        # מבצע-חסר: דירוג גבוה אך תפוקה נמוכה
        (s["overall"] >= 80) & (s["attacking_involvement_score"] < 50),
    ]
    # תיאור מילולי לכל תנאי
    choices = [
        "over-performer (hidden gem: output > rating)",
        "elite (extreme but consistent output)",
        "under-performer (rating > output)",
    ]
    # מקצים את הכיוון לפי התנאי המתקיים (ברירת מחדל: פרופיל חריג)
    anomalies["direction"] = np.select(conds, choices, default="unusual profile")

    # עמודות התצוגה
    cols = ["short_name", "position_group", "age", "overall",
            "attacking_involvement_score", "goals_per_match", "conversion_rate",
            "direction", "anomaly_score"]
    # ממיינים מהחריג ביותר ומחזירים top_n
    return anomalies.sort_values("anomaly_score")[cols].head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------
# Model comparison — Isolation Forest vs DBSCAN (vs One-Class SVM)
# ---------------------------------------------------------------------
# השוואת מודלים: כמה חריגות כל שיטה מצאה וכמה הן חופפות (לדרישת המחוון)
def compare_if_vs_dbscan(df, contamination=ANOMALY_CONTAMINATION):
    """Run IF, DBSCAN and One-Class SVM on the SAME bargain feature space and
    report how many anomalies each found and how much they overlap."""
    # עובדים על שחקנים עם שווי חיובי
    work = df[(df["value_eur"].notna()) & (df["value_eur"] > 0)].copy()
    # שווי בלוג
    work["log_value"] = np.log10(work["value_eur"])
    # מאגר ומטריצה מנורמלת
    pool, X = _scaled(work, BARGAIN_FEATURES)

    # חריגות לפי Isolation Forest
    if_anom, _ = run_isolation_forest(X, contamination)
    # חריגות לפי DBSCAN
    db_anom, _ = run_dbscan(X)
    # חריגות לפי One-Class SVM
    ocsvm_anom, _ = run_one_class_svm(X, nu=contamination)

    # קבוצות האינדקסים של החריגים בכל שיטה
    s_if = set(pool.index[if_anom])
    s_db = set(pool.index[db_anom])
    s_oc = set(pool.index[ocsvm_anom])
    # מחזירים את הספירות והחפיפות
    return {
        "isolation_forest": len(s_if),
        "dbscan": len(s_db),
        "one_class_svm": len(s_oc),
        "IF∩DBSCAN": len(s_if & s_db),
        "IF∩OCSVM": len(s_if & s_oc),
        "all_three": len(s_if & s_db & s_oc),
        "total_players": len(pool),
    }


# בלוק שמורץ בהרצה ישירה — מציאות, אי-התאמות, והשוואת מודלים
def main():
    # שורש הפרויקט
    root = Path(__file__).resolve().parent.parent
    # טוענים את הטבלה המרכזית
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    # כותרת
    print("=" * 70, "\nSTAGE 11 — Anomaly detection (USE #2)\n", "=" * 70)

    # [2a] מציאות עילית-וזולה (דירוג מינימלי 78)
    print("\n[2a] BARGAIN players, elite-but-cheap (min_overall=78):")
    b = detect_bargain_players(df, min_overall=78, top_n=8)
    print(b.to_string(index=False))
    # בדיקת איכות: כל הציונים חיוביים
    print("  quality: all market_efficiency_score > 0:",
          bool((b["market_efficiency_score"] > 0).all()))

    # [2b] אי-התאמות פרופיל-ביצועים (שחקנים עם נתוני אירועים)
    print("\n[2b] PROFILE-vs-PERFORMANCE anomalies (players with event data):")
    a = detect_profile_performance_anomalies(df, top_n=8)
    print(a.to_string(index=False))

    # כותרת השוואת המודלים
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Isolation Forest vs DBSCAN vs One-Class SVM")
    print("=" * 70)
    # מריצים את ההשוואה
    cmp = compare_if_vs_dbscan(df)
    # מדפיסים כל מדד
    for k, v in cmp.items():
        print(f"  {k:<16}: {v:,}")
    # פרשנות התוצאה
    print("Interpretation: IF flags a fixed ~2% by isolation depth; DBSCAN flags "
          "density-based noise (count varies with eps); their intersection are the "
          "most robust outliers. One-Class SVM gives a third boundary for comparison.")


# נקודת כניסה
if __name__ == "__main__":
    main()
