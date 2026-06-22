"""
data_loader.py — Loading functions for the two raw data sources.
Master Scout · Stage 2.

Functions:
    load_players_data(path, fifa_version=24)  -> FC24 player profiles
    load_events_data(path)                    -> Football Events match events
"""

# ייבוא Path לעבודה נוחה ובטוחה עם נתיבי קבצים בכל מערכת הפעלה
from pathlib import Path
# ייבוא pandas — ספריית טבלאות הנתונים שבה אנו טוענים ומעבדים את ה-CSV
import pandas as pd

# ---- Column selections (decided in Stage 1, docs/01_schema.md) ----

# רשימת העמודות שנשמור ממאגר השחקנים (FC24) — רק העמודות הרלוונטיות לפרויקט
PLAYER_COLUMNS = [
    # identity / meta
    # עמודות זהות ומטא: מזהה שחקן, גרסת FIFA, שמות, עמדות, גיל, תאריך לידה, לאום, מועדון, ליגה, רגל חזקה
    "player_id", "fifa_version", "short_name", "long_name", "player_positions",
    # המשך עמודות הזהות: גיל, תאריך לידה, נבחרת, מועדון, ליגה ורגל מועדפת
    "age", "dob", "nationality_name", "club_name", "league_name", "preferred_foot",
    # market / value
    # עמודות שוק וערך: שווי, שכר, סעיף שחרור, דירוג כללי ופוטנציאל
    "value_eur", "wage_eur", "release_clause_eur", "overall", "potential",
    # מוניטין בינלאומי של השחקן (1–5)
    "international_reputation",
    # physical
    # עמודות פיזיות: גובה, משקל, רגל חלשה, מהלכי מיומנות, קצב עבודה, מבנה גוף
    "height_cm", "weight_kg", "weak_foot", "skill_moves", "work_rate", "body_type",
    # core 6 face stats
    # 6 תכונות הליבה ("הפנים" של הכרטיס): מהירות, בעיטות, מסירות, כדרור, הגנה, פיזיות
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    # selected detailed attributes
    # תכונות התקפה מפורטות: גמר, מסירה קצרה, הרמות, נגיחות
    "attacking_finishing", "attacking_short_passing", "attacking_crossing",
    # תכונות מיומנות: נגיחה, כדרור, שליטת כדור, מסירה ארוכה, דיוק בעיטות חופשיות
    "attacking_heading_accuracy", "skill_dribbling", "skill_ball_control",
    # תכונות מיומנות ותנועה: מסירה ארוכה, דיוק בעיטה חופשית, תאוצה
    "skill_long_passing", "skill_fk_accuracy", "movement_acceleration",
    # תכונות תנועה: מהירות ריצה, זריזות, תגובות
    "movement_sprint_speed", "movement_agility", "movement_reactions",
    # תכונות עוצמה: כוח בעיטה, סיבולת, חוזק, בעיטות מרחוק
    "power_shot_power", "power_stamina", "power_strength", "power_long_shots",
    # תכונות מנטליות: ראיית משחק, מיקום, חטיפות, קור רוח
    "mentality_vision", "mentality_positioning", "mentality_interceptions",
    # תכונות מנטליות והגנה: קור רוח, חטיפה בעמידה, החלקה
    "mentality_composure", "defending_standing_tackle", "defending_sliding_tackle",
    # מודעות לסימון (הגנה אישית)
    "defending_marking_awareness",
]  # סוגר את רשימת עמודות השחקנים

# רשימת העמודות שנשמור ממאגר אירועי המשחק (Football Events)
EVENT_COLUMNS = [
    # מזהה משחק, דקה, טקסט תיאור, סוג אירוע, סוג אירוע משני, צד (בית/חוץ)
    "id_odsp", "time", "text", "event_type", "event_type2", "side",
    # קבוצת האירוע, היריבה, השחקן המעורב, שחקן משני, מיקום הבעיטה, תוצאת הבעיטה
    "event_team", "opponent", "player", "player2", "shot_place", "shot_outcome",
    # האם גול, מיקום במגרש, חלק גוף, שיטת הבישול, סיטואציה, התקפת מעבר
    "is_goal", "location", "bodypart", "assist_method", "situation", "fast_break",
]  # סוגר את רשימת עמודות האירועים


# פונקציה לטעינת פרופילי השחקנים מקובץ ה-CSV, עם סינון לגרסת FIFA מבוקשת
def load_players_data(path, fifa_version=24):
    """Load FC player profiles. By default keeps only the latest edition (FC24).

    Args:
        path: path to male_players.csv
        fifa_version: which FIFA edition to keep (default 24). Pass None to keep all.

    Returns:
        DataFrame, one row per player (for the chosen version).
    """
    # קוראים את ה-CSV ושומרים רק את העמודות שהוגדרו ב-PLAYER_COLUMNS
    df = pd.read_csv(path, low_memory=False, usecols=PLAYER_COLUMNS)
    # אם התבקשה גרסת FIFA ספציפית (ברירת מחדל 24) — נסנן אליה בלבד
    if fifa_version is not None:
        # מסננים את הטבלה לשורות של אותה גרסת FIFA ומעתיקים כדי לא לפגוע במקור
        df = df[df["fifa_version"] == fifa_version].copy()
        # within one edition a player_id is unique; guard anyway
        # מסירים כפילויות לפי מזהה שחקן (שמירה מפני שורות כפולות) ומאפסים אינדקס
        df = df.drop_duplicates(subset="player_id", keep="first").reset_index(drop=True)
    # מחזירים את טבלת השחקנים המסוננת
    return df


# פונקציה לטעינת אירועי המשחק מקובץ ה-CSV
def load_events_data(path):
    """Load Football Events match events.

    Args:
        path: path to events.csv

    Returns:
        DataFrame, one row per match event.
    """
    # קוראים את ה-CSV של האירועים ושומרים רק את העמודות שהוגדרו ב-EVENT_COLUMNS
    df = pd.read_csv(path, low_memory=False, usecols=EVENT_COLUMNS)
    # מחזירים את טבלת האירועים
    return df


# בלוק שמורץ רק כשמריצים את הקובץ ישירות (בדיקת שפיות מהירה)
if __name__ == "__main__":
    # מחשבים את תיקיית השורש של הפרויקט (שתי רמות מעל קובץ זה)
    here = Path(__file__).resolve().parent.parent
    # טוענים את מאגר השחקנים מתיקיית הנתונים הגולמיים
    players = load_players_data(here / "data/raw/male_players.csv")
    # טוענים את מאגר האירועים מתיקיית הנתונים הגולמיים
    events = load_events_data(here / "data/raw/events.csv")
    # מדפיסים את ממדי טבלת השחקנים (שורות × עמודות) לאחר הסינון ל-FC24
    print("players (FC24):", players.shape)
    # מדפיסים את ממדי טבלת האירועים
    print("events:", events.shape)
