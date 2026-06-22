"""
run_eda.py — Stage 2 EDA.
Produces descriptive statistics, missing-value percentages, and charts.

Outputs:
    reports/figures/*.png   (6 charts)
    docs/02_eda.md          (written by the caller / printed summary)

Run:
    conda activate masterscout
    python scripts/run_eda.py
"""

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא sys כדי להוסיף את src לנתיב
import sys

# ייבוא numpy לחישובים מספריים
import numpy as np
# ייבוא pandas לעבודה עם הטבלאות
import pandas as pd
# ייבוא matplotlib לציור גרפים
import matplotlib
matplotlib.use("Agg")  # headless, no display needed
# ייבוא ממשק ה-pyplot
import matplotlib.pyplot as plt
# ייבוא seaborn לגרפים יפים יותר
import seaborn as sns

# שורש הפרויקט
ROOT = Path(__file__).resolve().parent.parent
# מוסיפים את src לנתיב הייבוא
sys.path.insert(0, str(ROOT / "src"))
# מייבאים את פונקציות טעינת הנתונים
from data_loader import load_players_data, load_events_data  # noqa: E402

# תיקיית הגרפים
FIG_DIR = ROOT / "reports" / "figures"
# יוצרים את תיקיית הגרפים אם אינה קיימת
FIG_DIR.mkdir(parents=True, exist_ok=True)
# מגדירים ערכת עיצוב ל-seaborn
sns.set_theme(style="whitegrid")

# Event-type code -> name (from docs/01_schema.md), for the events chart
# מיפוי קוד סוג-אירוע לשם — עבור גרף האירועים
EVENT_TYPE_NAMES = {
    1: "Attempt", 2: "Corner", 3: "Foul", 4: "Yellow card",
    5: "2nd yellow", 6: "Red card", 7: "Substitution", 8: "Free kick won",
    9: "Offside", 10: "Hand ball", 11: "Penalty conceded",
}


# פונקציה: מחזירה דוח ערכים חסרים (כמות ואחוז) לכל עמודה
def missing_report(df, name):
    """Return a DataFrame of missing-value count and percentage per column."""
    # כמות הערכים החסרים בכל עמודה
    miss = df.isna().sum()
    # אחוז הערכים החסרים
    pct = (miss / len(df) * 100).round(2)
    # מרכיבים טבלת דוח
    rep = pd.DataFrame({"missing": miss, "missing_pct": pct})
    # שומרים רק עמודות עם חוסרים וממיינים
    rep = rep[rep["missing"] > 0].sort_values("missing_pct", ascending=False)
    # כותרת
    print(f"\n--- Missing values: {name} ({len(df):,} rows) ---")
    # אם אין חוסרים — מודיעים
    if rep.empty:
        print("  (no missing values)")
    else:
        # אחרת מדפיסים את הדוח
        print(rep.to_string())
    # מחזירים את הדוח
    return rep


# פונקציה: מדפיסה סטטיסטיקה תיאורית (ממוצע/חציון/סטיית תקן/מין/מקס)
def describe_numeric(df, cols, name):
    """Print mean/median/std/min/max for key numeric columns."""
    # שומרים רק עמודות שקיימות
    present = [c for c in cols if c in df.columns]
    # מחשבים סטטיסטיקה ובוחרים את העמודות הרצויות
    desc = df[present].describe().T[["mean", "50%", "std", "min", "max"]]
    # משנים שם של 50% לחציון ומעגלים
    desc = desc.rename(columns={"50%": "median"}).round(2)
    # כותרת
    print(f"\n--- Descriptive stats: {name} ---")
    # מדפיסים את הטבלה
    print(desc.to_string())
    # מחזירים אותה
    return desc


# בלוק שמורץ בהרצה ישירה — מריץ את כל ניתוח ה-EDA ומייצר גרפים
def main():
    # כותרת
    print("=" * 70)
    print("STAGE 2 — EDA")
    print("=" * 70)

    # טוענים את שחקני FC24
    players = load_players_data(ROOT / "data/raw/male_players.csv")   # FC24 only
    # טוענים את האירועים
    events = load_events_data(ROOT / "data/raw/events.csv")

    # מדפיסים את ממדי טבלת השחקנים
    print(f"\nplayers (FC24): {players.shape[0]:,} rows x {players.shape[1]} cols")
    # מדפיסים את ממדי טבלת האירועים
    print(f"events:         {events.shape[0]:,} rows x {events.shape[1]} cols")

    # ---------- Descriptive statistics ----------
    # סטטיסטיקה תיאורית לעמודות המספריות המרכזיות של השחקנים
    describe_numeric(
        players,
        ["age", "overall", "potential", "value_eur", "wage_eur",
         "pace", "shooting", "passing", "dribbling", "defending", "physic"],
        "FC24 players",
    )
    # סטטיסטיקה תיאורית לאירועים
    describe_numeric(events, ["time", "is_goal"], "events")

    # ---------- Missing values ----------
    # דוח ערכים חסרים לשחקנים
    missing_report(players, "FC24 players")
    # דוח ערכים חסרים לאירועים
    missing_report(events, "events")

    # ---------- Extra context ----------
    # הקשר נוסף: גולים וסיכומים קטגוריאליים
    print("\n--- Events: goals & key categorical sums ---")
    # סך הגולים
    print(f"  total goals (is_goal==1): {int(events['is_goal'].sum()):,}")
    # מספר המשחקים הייחודיים
    print(f"  unique matches:           {events['id_odsp'].nunique():,}")
    # מספר השחקנים הייחודיים
    print(f"  unique players:           {events['player'].nunique():,}")

    # ====================================================================
    # CHARTS  (saved as PNG to reports/figures/)
    # ====================================================================

    # 1. Age distribution
    # גרף 1: התפלגות גילאים
    plt.figure(figsize=(8, 5))
    sns.histplot(players["age"].dropna(), bins=30, color="#2a9d8f")
    plt.title("FC24 — Age distribution")
    plt.xlabel("Age"); plt.ylabel("Number of players")
    plt.tight_layout(); plt.savefig(FIG_DIR / "01_age_distribution.png", dpi=120); plt.close()

    # 2. Overall rating distribution
    # גרף 2: התפלגות הדירוג הכללי
    plt.figure(figsize=(8, 5))
    sns.histplot(players["overall"].dropna(), bins=40, color="#264653")
    plt.title("FC24 — Overall rating distribution")
    plt.xlabel("Overall"); plt.ylabel("Number of players")
    plt.tight_layout(); plt.savefig(FIG_DIR / "02_overall_distribution.png", dpi=120); plt.close()

    # 3. Market value distribution (log scale — heavily right-skewed)
    # גרף 3: התפלגות שווי השוק (סקאלה לוגריתמית כי מוטה ימינה)
    val = players["value_eur"].dropna()
    # שומרים רק ערכים חיוביים (ללוג)
    val = val[val > 0]
    plt.figure(figsize=(8, 5))
    sns.histplot(val, bins=50, color="#e76f51", log_scale=(True, False))
    plt.title("FC24 — Market value distribution (log scale)")
    plt.xlabel("Value (EUR, log)"); plt.ylabel("Number of players")
    plt.tight_layout(); plt.savefig(FIG_DIR / "03_value_distribution.png", dpi=120); plt.close()

    # 4. Event counts by type
    # גרף 4: ספירת אירועים לפי סוג
    counts = events["event_type"].map(EVENT_TYPE_NAMES).value_counts()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#457b9d")
    plt.title("Football Events — Event counts by type")
    plt.xlabel("Count"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(FIG_DIR / "04_event_counts.png", dpi=120); plt.close()

    # 5. (bonus) Overall vs Value — shows the value premium for top players
    # גרף 5: דירוג מול שווי — מראה את פרמיית השווי לשחקני העילית
    plt.figure(figsize=(8, 5))
    # דוגמים עד 4000 שחקנים לתצוגה
    sample = players[(players["value_eur"] > 0)].sample(min(4000, len(players)), random_state=42)
    sns.scatterplot(data=sample, x="overall", y="value_eur", alpha=0.3, color="#6a4c93")
    plt.yscale("log")
    plt.title("FC24 — Overall vs Market value (sample)")
    plt.xlabel("Overall"); plt.ylabel("Value (EUR, log)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "05_overall_vs_value.png", dpi=120); plt.close()

    # 6. (bonus) Bodypart of shots — relevant to the two-footed feature later
    # גרף 6: בעיטות לפי חלק גוף — בסיס לתכונת הדו-רגליות
    shots = events[events["event_type"] == 1]
    # מיפוי חלק גוף לשם
    bp_names = {1: "Right foot", 2: "Left foot", 3: "Head"}
    bp = shots["bodypart"].map(bp_names).value_counts()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=bp.index, y=bp.values, color="#f4a261")
    plt.title("Football Events — Shots by body part")
    plt.xlabel(""); plt.ylabel("Number of shots")
    plt.tight_layout(); plt.savefig(FIG_DIR / "06_shots_by_bodypart.png", dpi=120); plt.close()

    # מדפיסים שנשמרו 6 גרפים
    print(f"\nSaved 6 charts to {FIG_DIR}")
    # סיום
    print("\nSTAGE 2 EDA complete.")


# נקודת כניסה
if __name__ == "__main__":
    main()
