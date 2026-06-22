"""
player_match_stats.py — Stage 5.
Aggregate clean_events to the player-in-a-single-match level.

Output: data/processed/player_match_stats.csv  (one row per player x match)

Grouping key: clean_name + id_odsp  (a player's line in one specific match).
This table is the basis for "braces" (matches with 2+ goals) in Stage 6.
No numeric thresholds here — pure aggregation of the binary columns from Stage 4.
"""

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא sys (זמין במידת הצורך לנתיבים)
import sys

# ייבוא pandas לעבודה עם הטבלאות
import pandas as pd

# binary columns produced in Stage 4 that we sum up per player-match
# מיפוי עמודות הדגל (משלב 4) לשמות הסיכום שלהן לרמת שחקן×משחק
SUM_COLUMNS = {
    # בעיטות
    "is_shot": "shots",
    # גולים
    "is_goal": "goals",
    # מסירות מפתח
    "is_key_pass": "key_passes",
    # בעיטות מהרחבה
    "is_box_shot": "box_shots",
    # בעיטות למסגרת
    "is_on_target": "shots_on_target",
    # בעיטות ברגל שמאל
    "is_left_foot": "left_foot_shots",
    # בעיטות ברגל ימין
    "is_right_foot": "right_foot_shots",
    # נגיחות
    "is_header": "header_shots",
    # כרטיסים צהובים
    "is_yellow": "yellow_cards",
    # כרטיסים אדומים
    "is_red": "red_cards",
    # עבירות
    "is_foul": "fouls",
    # בישולים ממסירת עומק
    "is_through_ball_assist": "through_ball_assists",
}


# פונקציה שמכווצת אירועים לשורה אחת לכל (שחקן, משחק)
def build_player_match_stats(clean_events_df):
    """Aggregate events to one row per (clean_name, id_odsp).

    Drops team-level events with no player (clean_name is NaN).
    """
    # מסירים אירועים בלי שם שחקן (אירועים קבוצתיים) ועובדים על עותק
    df = clean_events_df.dropna(subset=["clean_name"]).copy()

    # מקבצים לפי שחקן+משחק ומסכמים את כל עמודות הדגל, ואז משנים שם לעמודות הסיכום
    agg = (
        df.groupby(["clean_name", "id_odsp"], as_index=False)[list(SUM_COLUMNS)]
        .sum()
        .rename(columns=SUM_COLUMNS)
    )

    # carry the team the player appeared for in that match (most frequent)
    # מצרפים את הקבוצה שבה שיחק השחקן באותו משחק (הערך השכיח ביותר)
    team = (
        df.groupby(["clean_name", "id_odsp"])["event_team"]
        # בוחרים את שם הקבוצה השכיח; אם אין — None
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else None)
        .reset_index()
        .rename(columns={"event_team": "match_team"})
    )
    # ממזגים את הקבוצה אל טבלת הסיכומים לפי שחקן+משחק
    out = agg.merge(team, on=["clean_name", "id_odsp"], how="left")

    # מחזירים את הטבלה ברמת שחקן×משחק
    return out


# בלוק שמורץ בהרצה ישירה — בונה ומאמת את טבלת שחקן×משחק
def main():
    # שורש הפרויקט
    root = Path(__file__).resolve().parent.parent
    # מקור הנתונים: טבלת האירועים הנקייה
    src = root / "data/processed/clean_events.csv"

    # only load the columns we need (the file is large)
    # טוענים רק את העמודות הדרושות (הקובץ גדול מאוד)
    usecols = ["clean_name", "id_odsp", "event_team"] + list(SUM_COLUMNS)
    # קריאת ה-CSV עם העמודות הנבחרות
    events = pd.read_csv(src, usecols=usecols, low_memory=False)

    # בונים את טבלת שחקן×משחק
    pms = build_player_match_stats(events)

    # נתיב הפלט
    out = root / "data/processed/player_match_stats.csv"
    # שמירה ל-CSV
    pms.to_csv(out, index=False)

    # ---- quality checks ----
    # ממדי הטבלה ושם הקובץ
    print(f"player_match_stats: {pms.shape[0]:,} rows x {pms.shape[1]} cols -> {out.name}")
    # מספר השחקנים הייחודיים
    print(f"unique players: {pms['clean_name'].nunique():,}")
    # מספר המשחקים הייחודיים
    print(f"unique matches: {pms['id_odsp'].nunique():,}")
    # אימות סך הגולים מול הידוע (24,446)
    print(f"total goals (reconcile vs clean_events 24,446): {int(pms['goals'].sum()):,}")
    # מקסימום גולים של שחקן במשחק בודד
    print(f"max goals by a player in one match: {int(pms['goals'].max())}")
    # מקרים שבהם גולים > בעיטות (פנדלים/גול עצמי יכולים לחרוג מהבעיטות שנספרו)
    print(f"rows where goals > shots: {(pms['goals'] > pms['shots']).sum()}  "
          "(penalties/own-goals can exceed counted shots)")
    # מקרים שבהם בעיטות-רחבה > בעיטות (מצופה 0)
    print(f"rows where box_shots > shots: {(pms['box_shots'] > pms['shots']).sum()}  (should be 0)")
    # בדיקת שפיות לדאבלים — משחקים עם 2+ גולים
    print("\nmatches with 2+ goals (braces) — sanity, top scorers' big games:")
    # מסננים למשחקים עם 2+ גולים
    braces = pms[pms["goals"] >= 2]
    # מספר משחקי הדאבל הכולל
    print(f"  total brace performances: {len(braces):,}")
    # 5 המשחקים עם הכי הרבה גולים
    print(braces.nlargest(5, "goals")[["clean_name", "id_odsp", "goals", "shots"]].to_string(index=False))


# נקודת כניסה
if __name__ == "__main__":
    main()
