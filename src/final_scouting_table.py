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

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא pandas לטעינה, מיזוג ושמירה של הטבלאות
import pandas as pd

# event-derived columns to bring across (the rest stay in player_event_stats.csv)
# רשימת העמודות מנתוני האירועים שנמזג אל הטבלה המרכזית (השאר נשארות בטבלת האירועים)
EVENT_COLUMNS = [
    # סיכומים עונתיים: משחקים, גולים, בעיטות, מסירות מפתח
    "matches", "total_goals", "total_shots", "total_key_passes",
    # סיכומים נוספים: בעיטות מהרחבה, כרטיסים צהובים, כרטיסים אדומים
    "total_box_shots", "total_yellow_cards", "total_red_cards",
    # מספר משחקי הדאבל (2+ גולים באותו משחק)
    "matches_with_2_plus_goals",
    # ממוצעים למשחק: גולים, בעיטות, מסירות מפתח
    "goals_per_match", "shots_per_match", "key_passes_per_match",
    # שיעורים (0–1): דיוק בעיטות, אחוז המרה, אחוז בעיטות מהרחבה
    "shot_accuracy", "conversion_rate", "box_shot_rate",
    # ציונים מחושבים: מעורבות התקפית, יצירתיות
    "attacking_involvement_score", "creative_score",
    # ציונים מחושבים: משמעת, איזון דו-רגלי
    "discipline_score", "foot_balance_score",
]  # סוגר את רשימת העמודות מהאירועים


# פונקציה הבונה את הטבלה המרכזית: מיזוג שמאלי של שחקני FC24 עם נתוני האירועים
def build_final_scouting_table(players, event_stats):
    """Left-join FC24 players with event stats on clean_name; add source flags."""
    # event_stats is keyed by clean_name (unique). Keep only the columns we merge.
    # שומרים מטבלת האירועים רק את מפתח ההתאמה והעמודות הרצויות
    ev = event_stats[["clean_name"] + EVENT_COLUMNS].copy()

    # מיזוג שמאלי לפי clean_name: כל שחקני FC24 נשמרים; validate מוודא יחס רבים-לאחד
    merged = players.merge(ev, on="clean_name", how="left", validate="m:1")

    # source flags
    # דגל has_event_data = האם נמצאו לשחקן נתוני אירועים (matches לא ריק)
    merged["has_event_data"] = merged["matches"].notna()
    # הערת מקור טקסטואלית בהתאם לקיום נתוני האירועים
    merged["data_source_note"] = merged["has_event_data"].map({
        # יש התאמה: פרופיל FC24 + ביצועי אירועים
        True: "FC24 profile + Football Events performance",
        # אין התאמה: רק פרופיל FC24 (פער שנים / חוסר התאמת שם)
        False: "FC24 profile only (no event-match found; year-gap/name-mismatch)",
    })
    # מחזירים את הטבלה הממוזגת
    return merged


# בלוק שמורץ רק בהרצה ישירה של הקובץ — בונה ומאמת את הטבלה המרכזית
def main():
    # מחשבים את שורש הפרויקט (שתי רמות מעל קובץ זה)
    root = Path(__file__).resolve().parent.parent
    # טוענים את טבלת השחקנים הנקייה
    players = pd.read_csv(root / "data/processed/clean_players.csv")
    # טוענים את טבלת ציוני האירועים לרמת שחקן
    event_stats = pd.read_csv(root / "data/processed/player_event_stats.csv")

    # בונים את הטבלה המרכזית באמצעות פונקציית המיזוג
    final = build_final_scouting_table(players, event_stats)
    # נתיב הפלט של הטבלה המרכזית
    out = root / "data/processed/final_scouting_table.csv"
    # שומרים את הטבלה ל-CSV בלי עמודת אינדקס
    final.to_csv(out, index=False)

    # ---- quality checks ----
    # מספר השחקנים הכולל בטבלה
    n = len(final)
    # כמה שחקנים יש להם נתוני אירועים אמיתיים
    matched = int(final["has_event_data"].sum())
    # מדפיסים את ממדי הטבלה ושם קובץ הפלט
    print(f"final_scouting_table: {n:,} rows x {final.shape[1]} cols -> {out.name}")
    # מאמתים שאין מזהי שחקן כפולים (מצופה 0)
    print(f"duplicate player_id: {int(final['player_id'].duplicated().sum())}  (expect 0)")
    # אחוז השחקנים עם נתוני אירועים
    print(f"has_event_data = True:  {matched:,}  ({matched/n*100:.1f}%)")
    # אחוז השחקנים ללא נתוני אירועים
    print(f"has_event_data = False: {n-matched:,}  ({(n-matched)/n*100:.1f}%)")

    # name collisions: FC24 players sharing a clean_name (ambiguous merge)
    # זיהוי התנגשויות שם: שחקני FC24 בעלי אותו clean_name (מיזוג עמום)
    dup_names = final["clean_name"].duplicated(keep=False) & final["clean_name"].notna()
    # מדפיסים כמה שמות כפולים נמצאו
    print(f"FC24 players sharing a clean_name (ambiguous): {int(dup_names.sum())}")

    # coverage by position group
    # מדפיסים כותרת: כיסוי נתוני אירועים לפי קבוצת עמדה
    print("\nEvent-data coverage by position_group:")
    # מקבצים לפי עמדה ומסכמים כמה יש נתונים מתוך כמה שחקנים
    cov = final.groupby("position_group")["has_event_data"].agg(["sum", "count"])
    # מוסיפים עמודת אחוז כיסוי
    cov["pct"] = (cov["sum"] / cov["count"] * 100).round(1)
    # מדפיסים את טבלת הכיסוי
    print(cov.to_string())

    # sanity: a few well-known players that SHOULD match
    # בדיקת שפיות: כמה שמות מוכרים שאמורים להימצא עם נתוני אירועים
    print("\nSpot check — top FC24 names with event data:")
    # עמודות לתצוגה בבדיקת השפיות
    cols = ["short_name", "clean_name", "overall", "has_event_data",
            "matches", "total_goals"]
    # בוחרים 8 השחקנים בעלי הדירוג הגבוה ביותר מבין אלה עם נתוני אירועים
    have = final[final["has_event_data"]].nlargest(8, "overall")[cols]
    # מדפיסים אותם
    print(have.to_string(index=False))


# נקודת הכניסה: הרצה ישירה מפעילה את main
if __name__ == "__main__":
    main()
