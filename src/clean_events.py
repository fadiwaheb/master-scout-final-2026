"""
clean_events.py — Stage 4.
Clean the Football Events table, translate codes to text names, and build the
binary feature columns used later for player aggregation.

Output: data/processed/clean_events.csv  (one row per match event)

Code mappings come from docs/01_schema.md (verified against the raw values).
The one numeric THRESHOLD here is the set of "inside the box" location codes
(is_box_shot) — documented in docs/ספים_והחלטות_Master_Scout.docx.
"""

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא sys כדי להוסיף את תיקיית src לנתיב הייבוא
import sys

# ייבוא numpy לחישובים מספריים
import numpy as np
# ייבוא pandas לעבודה עם הטבלאות
import pandas as pd

# מוסיפים את תיקיית הקובץ הנוכחי לנתיב כדי לייבא מודולים אחים
sys.path.insert(0, str(Path(__file__).resolve().parent))
# מייבאים את אותה פונקציית נרמול שם כדי שמפתח ההתאמה יהיה זהה לזה של טבלת השחקנים
from clean_players import clean_player_name  # reuse the same normalization key

# =====================================================================
# CODE -> NAME mappings (from docs/01_schema.md)
# =====================================================================
# מילון תרגום קוד סוג-אירוע ראשי לשם קריא
EVENT_TYPE_NAMES = {
    # 1=ניסיון, 2=קרן, 3=עבירה, 4=כרטיס צהוב
    1: "Attempt", 2: "Corner", 3: "Foul", 4: "Yellow card",
    # 5=צהוב שני, 6=אדום, 7=חילוף
    5: "Second yellow card", 6: "Red card", 7: "Substitution",
    # 8=עבירה שהושגה, 9=נבדל, 10=יד, 11=פנדל שניתן
    8: "Free kick won", 9: "Offside", 10: "Hand ball", 11: "Penalty conceded",
}
# מילון תרגום קוד אירוע משני לשם קריא
EVENT_TYPE2_NAMES = {
    # 12=מסירת מפתח, 13=מסירת עומק כושלת, 14=הרחקה, 15=גול עצמי
    12: "Key pass", 13: "Failed through ball", 14: "Sending off", 15: "Own goal",
}
# מילון תרגום חלק הגוף שבו בוצעה הבעיטה
BODYPART_NAMES = {1: "Right foot", 2: "Left foot", 3: "Head"}
# מילון תרגום תוצאת הבעיטה
SHOT_OUTCOME_NAMES = {1: "On target", 2: "Off target", 3: "Blocked", 4: "Hit the bar"}
# מילון תרגום הסיטואציה (משחק פתוח / כדור נייח / קרן / בעיטה חופשית)
SITUATION_NAMES = {1: "Open play", 2: "Set piece", 3: "Corner", 4: "Free kick"}
# מילון תרגום שיטת הבישול
ASSIST_METHOD_NAMES = {0: "None", 1: "Pass", 2: "Cross", 3: "Headed pass", 4: "Through ball"}
# מילון תרגום מיקום הבעיטה במגרש
LOCATION_NAMES = {
    # 1=מחצית התקפית, 2=מחצית הגנתית, 3=מרכז הרחבה
    1: "Attacking half", 2: "Defensive half", 3: "Centre of the box",
    # 4=אגף שמאל, 5=אגף ימין, 6=זווית קשה וטווח ארוך
    4: "Left wing", 5: "Right wing", 6: "Difficult angle and long range",
    # 7=זווית קשה משמאל, 8=זווית קשה מימין
    7: "Difficult angle on the left", 8: "Difficult angle on the right",
    # 9=צד שמאל של הרחבה, 10=צד שמאל של רחבת ה-5
    9: "Left side of the box", 10: "Left side of the six yard box",
    # 11=צד ימין של הרחבה, 12=צד ימין של רחבת ה-5
    11: "Right side of the box", 12: "Right side of the six yard box",
    # 13=טווח קצר מאוד, 14=נקודת הפנדל, 15=מחוץ לרחבה
    13: "Very close range", 14: "Penalty spot", 15: "Outside the box",
    # 16=טווח ארוך, 17=מעל 35 יארד, 18=מעל 40 יארד
    16: "Long range", 17: "More than 35 yards", 18: "More than 40 yards",
    # 19=לא תועד
    19: "Not recorded",
}

# =====================================================================
# THRESHOLD: which location codes count as "inside the box"
# Reasoning: a shot from inside the penalty area is a high-quality chance.
# These 7 codes are the locations geometrically inside the box (the six-yard
# box, the sides of the box, the centre, very close range, and the penalty spot).
# =====================================================================
# קבוצת קודי המיקום שנחשבים "בתוך הרחבה" — בסיס לדגל is_box_shot
IN_BOX_LOCATIONS = {3, 9, 10, 11, 12, 13, 14}


# פונקציה שמוסיפה עמודות שם קריאות ועמודות דגל בינאריות (0/1)
def map_event_codes(df):
    """Add readable *_name columns and binary is_* feature columns."""
    # עובדים על עותק כדי לא לשנות את הטבלה המקורית
    df = df.copy()

    # ---- text name columns ----
    # ממירים את קוד סוג-האירוע לשם קריא
    df["event_type_name"] = df["event_type"].map(EVENT_TYPE_NAMES)
    # ממירים את קוד האירוע המשני לשם קריא
    df["event_type2_name"] = df["event_type2"].map(EVENT_TYPE2_NAMES)
    # ממירים את קוד חלק הגוף לשם קריא
    df["bodypart_name"] = df["bodypart"].map(BODYPART_NAMES)
    # ממירים את קוד תוצאת הבעיטה לשם קריא
    df["shot_outcome_name"] = df["shot_outcome"].map(SHOT_OUTCOME_NAMES)
    # ממירים את קוד הסיטואציה לשם קריא
    df["situation_name"] = df["situation"].map(SITUATION_NAMES)
    # ממירים את קוד שיטת הבישול לשם קריא
    df["assist_method_name"] = df["assist_method"].map(ASSIST_METHOD_NAMES)
    # ממירים את קוד המיקום לשם קריא
    df["location_name"] = df["location"].map(LOCATION_NAMES)

    # ---- binary feature columns (0/1 int) ----
    # דגל: האם האירוע הוא בעיטה (סוג 1)
    df["is_shot"] = (df["event_type"] == 1).astype(int)
    # is_goal already exists in the raw file (0/1) — keep as int
    # דגל גול כבר קיים במקור — ממלאים ריקים ב-0 וממירים למספר שלם
    df["is_goal"] = df["is_goal"].fillna(0).astype(int)
    # דגל: האם זו מסירת מפתח (אירוע משני 12)
    df["is_key_pass"] = (df["event_type2"] == 12).astype(int)
    # דגל: האם הבעיטה מתוך הרחבה (לפי קבוצת קודי המיקום)
    df["is_box_shot"] = df["location"].isin(IN_BOX_LOCATIONS).astype(int)
    # דגל: בעיטה ברגל שמאל (חלק גוף 2)
    df["is_left_foot"] = (df["bodypart"] == 2).astype(int)
    # דגל: בעיטה ברגל ימין (חלק גוף 1)
    df["is_right_foot"] = (df["bodypart"] == 1).astype(int)
    # דגל: נגיחה (חלק גוף 3)
    df["is_header"] = (df["bodypart"] == 3).astype(int)
    # דגל: בעיטה למסגרת (תוצאה 1)
    df["is_on_target"] = (df["shot_outcome"] == 1).astype(int)
    # דגל: כרטיס צהוב (סוג 4)
    df["is_yellow"] = (df["event_type"] == 4).astype(int)
    # דגל: כרטיס אדום (סוג 5 או 6)
    df["is_red"] = (df["event_type"].isin([5, 6])).astype(int)
    # דגל: עבירה (סוג 3)
    df["is_foul"] = (df["event_type"] == 3).astype(int)
    # דגל: בישול ממסירת עומק (שיטת בישול 4)
    df["is_through_ball_assist"] = (df["assist_method"] == 4).astype(int)

    # מחזירים את הטבלה המועשרת
    return df


# פונקציה שמנקה את טבלת האירועים ומוסיפה את מפתח התאמת השחקן
def clean_events_data(df):
    """Clean the events table and attach the player matching key.

    Steps:
      1. translate codes + build binary columns (map_event_codes)
      2. add clean_name (normalized player name, for joining with players)
      3. keep all events; rows without a `player` get clean_name = NaN
         (they are legitimate team-level events such as some corners).
    """
    # מתרגמים קודים ובונים את עמודות הדגל
    df = map_event_codes(df)
    # מוסיפים שם מנורמל להתאמה מול טבלת השחקנים
    df["clean_name"] = df["player"].apply(clean_player_name)
    # מאפסים אינדקס ומחזירים
    return df.reset_index(drop=True)


# בלוק שמורץ בהרצה ישירה — בונה ומאמת את טבלת האירועים הנקייה
def main():
    # שורש הפרויקט (שתי רמות מעל הקובץ)
    root = Path(__file__).resolve().parent.parent
    # מוסיפים את src לנתיב הייבוא
    sys.path.insert(0, str(root / "src"))
    # מייבאים את פונקציית טעינת האירועים
    from data_loader import load_events_data

    # טוענים את האירועים הגולמיים
    raw = load_events_data(root / "data/raw/events.csv")
    # מנקים אותם
    clean = clean_events_data(raw)

    # נתיב הפלט של טבלת האירועים הנקייה
    out = root / "data/processed/clean_events.csv"
    # שומרים ל-CSV בלי עמודת אינדקס
    clean.to_csv(out, index=False)

    # ---- quality checks ----
    # מדפיסים את ממדי הטבלה ושם קובץ הפלט
    print(f"clean_events: {clean.shape[0]:,} rows x {clean.shape[1]} cols -> {out.name}")
    # מאמתים שסכום הגולים תואם את הידוע (24,446)
    print(f"is_goal sum:        {clean['is_goal'].sum():,}  (raw had 24,446)")
    # מדפיסים את מספר הבעיטות
    print(f"is_shot sum:        {clean['is_shot'].sum():,}")
    # מדפיסים את מספר מסירות המפתח
    print(f"is_key_pass sum:    {clean['is_key_pass'].sum():,}")
    # מדפיסים את מספר הבעיטות מהרחבה
    print(f"is_box_shot sum:    {clean['is_box_shot'].sum():,}")

    # every event_type translated?
    # בודקים שכל סוג-אירוע קיים תורגם (אין שם חסר כשהקוד קיים)
    untranslated = clean.loc[clean["event_type"].notna() & clean["event_type_name"].isna()]
    # מדפיסים כמה שורות לא תורגמו (מצופה 0)
    print(f"untranslated event_type rows: {len(untranslated)}")

    # do goals always have a player?
    # בודקים אם יש גולים בלי שם שחקן
    goals_no_player = clean.loc[(clean["is_goal"] == 1) & clean["clean_name"].isna()]
    # מדפיסים את מספרם
    print(f"goals with empty player: {len(goals_no_player)}")

    # box shots should be a subset of shots
    # בעיטות מהרחבה צריכות להיות תת-קבוצה של בעיטות
    box_not_shot = clean.loc[(clean["is_box_shot"] == 1) & (clean["is_shot"] == 0)]
    # מדפיסים כמה בעיטות-רחבה אינן בעיטות (מצופה 0)
    print(f"box_shots that are not shots: {len(box_not_shot)}")

    # מדפיסים פילוח גולים לפי חלק גוף לבדיקת שפיות
    print("\nGoals by body part (sanity):")
    # ספירת ערכים של חלק הגוף עבור הגולים בלבד
    print(clean.loc[clean["is_goal"] == 1, "bodypart_name"].value_counts(dropna=False).to_string())


# נקודת כניסה: הרצה ישירה מפעילה את main
if __name__ == "__main__":
    main()
