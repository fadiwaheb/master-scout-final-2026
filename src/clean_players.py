"""
clean_players.py — Stage 3.
Clean the FC24 player table and add computed score columns.

Output: data/processed/clean_players.csv  (one row per player)

All THRESHOLDS / WEIGHTS are defined as named constants at the top and are
documented in docs/03_thresholds.md (the central thresholds document).
Player and club NAMES are kept in English; only a normalized matching key
(`clean_name`) is added for later joining with the events table.
"""

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא unicodedata לנרמול תווים (הסרת סימנים דיאקריטיים)
import unicodedata
# ייבוא re לביטויים רגולריים (ניקוי פיסוק ורווחים)
import re

# ייבוא numpy לערכי NaN וחישובים
import numpy as np
# ייבוא pandas לעבודה עם הטבלה
import pandas as pd

# =====================================================================
# THRESHOLDS & WEIGHTS  (see docs/03_thresholds.md for the reasoning)
# =====================================================================

# -- position_group: derived from the FIRST listed position --
# Reasoning: a player's primary position is the first token of player_positions.
# מילון הממפה כל עמדת FC לאחת מ-4 קבוצות העמדה (שוער/הגנה/קישור/חלוץ)
POSITION_TO_GROUP = {
    # שוער
    "GK": "GK",
    # בלמים ומגנים → הגנה
    "CB": "Defender", "RB": "Defender", "LB": "Defender",
    # מגני אגף → הגנה
    "RWB": "Defender", "LWB": "Defender",
    # קשרים אחורי/מרכזי/התקפי → קישור
    "CDM": "Midfielder", "CM": "Midfielder", "CAM": "Midfielder",
    # קשרי אגף → קישור
    "RM": "Midfielder", "LM": "Midfielder",
    # מקצים, חלוץ מדומה וחלוץ מרכזי → חלוץ
    "RW": "Forward", "LW": "Forward", "CF": "Forward", "ST": "Forward",
}

# -- ability_score weights per position group --
# Reasoning: each role is judged by the attributes that matter for that role.
# Weights sum to 1.0. Forwards reward shooting/pace; defenders reward defending/
# physic; midfielders reward passing/dribbling. GK has no face stats, so for GK
# we fall back to the curated `overall` rating.
# משקלים לחישוב ציון היכולת לפי עמדה (סכום כל קבוצה = 1.0)
ABILITY_WEIGHTS = {
    # חלוץ: דגש על בעיטות, מהירות וכדרור
    "Forward":    {"shooting": 0.30, "pace": 0.20, "dribbling": 0.20,
                   "passing": 0.10, "physic": 0.10, "defending": 0.10},
    # קשר: דגש על מסירות וכדרור
    "Midfielder": {"passing": 0.30, "dribbling": 0.25, "pace": 0.15,
                   "shooting": 0.10, "defending": 0.10, "physic": 0.10},
    # מגן: דגש על הגנה ופיזיות
    "Defender":   {"defending": 0.40, "physic": 0.25, "pace": 0.15,
                   "passing": 0.10, "dribbling": 0.05, "shooting": 0.05},
}
# רשימת 6 תכונות הליבה ("פני" הכרטיס)
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
# שווי מינימלי (€) כדי שיהיה אפשר לחשב ציון יעילות שוק משמעותי
MIN_VALUE_EUR = 10_000


# =====================================================================
# Helpers
# =====================================================================

# פונקציה שמנרמלת שם לצורך התאמה: אותיות קטנות, הסרת סימנים, רווחים מאוחדים
def clean_player_name(name):
    """Normalize a name into a matching key: lowercase, strip accents/diacritics,
    collapse whitespace. Used ONLY for joining with the events table — the
    display name stays in its original English form.

    'Mladen Petrić' -> 'mladen petric'
    """
    # אם הקלט אינו מחרוזת — אין שם להחזיר
    if not isinstance(name, str):
        return None
    # decompose accents and drop the combining marks
    # פירוק התווים לצורתם הבסיסית + סימני ניקוד נפרדים
    nfkd = unicodedata.normalize("NFKD", name)
    # השמטת סימני הניקוד המשולבים (נשארות אותיות ASCII בלבד)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    # אותיות קטנות והסרת רווחים בקצוות
    ascii_str = ascii_str.lower().strip()
    # החלפת כל מה שאינו אות/ספרה/רווח ברווח (ניקוי פיסוק)
    ascii_str = re.sub(r"[^a-z0-9 ]", " ", ascii_str)   # drop punctuation
    # איחוד רווחים כפולים לרווח יחיד
    ascii_str = re.sub(r"\s+", " ", ascii_str).strip()
    # מחזירים את השם המנורמל, או None אם נשאר ריק
    return ascii_str or None


# פונקציה שממפה את העמדה הראשית (הראשונה ברשימה) לאחת מ-4 קבוצות העמדה
def assign_position_group(player_positions):
    """Map the first listed FC position to one of GK/Defender/Midfielder/Forward."""
    # אם אין רשימת עמדות תקינה — אין קבוצה
    if not isinstance(player_positions, str) or not player_positions.strip():
        return None
    # לוקחים את העמדה הראשונה (לפני הפסיק) ומנקים אותה לאותיות גדולות
    primary = player_positions.split(",")[0].strip().upper()
    # מחזירים את קבוצת העמדה מהמילון (או None אם לא נמצאה)
    return POSITION_TO_GROUP.get(primary)


# פונקציה שמחשבת ציון יכולת 0–100 לשחקן לפי משקלי העמדה שלו
def calculate_ability_score(row):
    """Position-weighted blend of the 6 face stats (0-100).
    GK -> use `overall` (no face stats available)."""
    # קבוצת העמדה של השחקן
    grp = row.get("position_group")
    # לשוער (או חסר עמדה) אין תכונות ליבה — נשתמש בדירוג הכללי
    if grp == "GK" or grp is None:
        return float(row["overall"]) if pd.notna(row.get("overall")) else np.nan
    # המשקלים המתאימים לקבוצת העמדה
    weights = ABILITY_WEIGHTS[grp]
    # אתחול סכום משוקלל וסכום משקלים בפועל
    total, wsum = 0.0, 0.0
    # עוברים על כל תכונה ומשקלה
    for stat, w in weights.items():
        # ערך התכונה אצל השחקן
        val = row.get(stat)
        # מוסיפים לסכום רק אם הערך קיים
        if pd.notna(val):
            total += w * float(val)
            wsum += w
    # אם אף תכונה לא הייתה זמינה — מחזירים NaN
    if wsum == 0:
        return np.nan
    # מנרמלים מחדש למקרה שחלק מהתכונות חסרו, ומעגלים
    return round(total / wsum, 2)   # renormalize if some stat was missing


# פונקציה וקטורית: ציון יעילות שוק = אחוזון יכולת פחות אחוזון שווי
def calculate_market_efficiency_score(df):
    """Vectorized: ability percentile minus value percentile, in [-100, 100].
    Higher = more underpriced relative to ability (candidate bargain)."""
    # רק שחקנים ששווים מעל הסף נחשבים זכאים לציון
    eligible = df["value_eur"].fillna(0) >= MIN_VALUE_EUR
    # אחוזון היכולת (0–100)
    ability_pct = df["ability_score"].rank(pct=True) * 100
    # לוג של השווי (כי השווי מוטה ימינה קיצונית), רק לזכאים
    log_value = np.log10(df["value_eur"].where(eligible))
    # אחוזון השווי (0–100)
    value_pct = log_value.rank(pct=True) * 100
    # ההפרש: ככל שגבוה יותר → השחקן "זול" יותר יחסית ליכולתו
    score = (ability_pct - value_pct).round(2)
    # ללא-זכאים מקבלים NaN
    score[~eligible] = np.nan
    # מחזירים את הציון
    return score


# =====================================================================
# Main cleaning pipeline
# =====================================================================

# פונקציה שמנקה את טבלת FC24 ומוסיפה עמודות מחושבות
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
    # מסירים כפילויות לפי מזהה שחקן ועובדים על עותק
    df = df.drop_duplicates(subset="player_id", keep="first").copy()

    # position group
    # מחשבים קבוצת עמדה מהעמדה הראשית
    df["position_group"] = df["player_positions"].apply(assign_position_group)

    # matching key (display name stays English in long_name/short_name)
    # מפתח התאמה מנורמל (שם התצוגה נשאר באנגלית)
    df["clean_name"] = df["long_name"].apply(clean_player_name)

    # scores
    # ציון יכולת משוקלל לפי עמדה (חישוב לכל שורה)
    df["ability_score"] = df.apply(calculate_ability_score, axis=1)
    # פוטנציאל גדילה = פוטנציאל פחות דירוג כללי
    df["potential_growth"] = (df["potential"] - df["overall"]).astype("Float64")
    # ציון יעילות שוק (מציאות)
    df["market_efficiency_score"] = calculate_market_efficiency_score(df)

    # מאפסים אינדקס ומחזירים
    return df.reset_index(drop=True)


# בלוק שמורץ בהרצה ישירה — בונה ומאמת את טבלת השחקנים הנקייה
def main():
    # שורש הפרויקט
    root = Path(__file__).resolve().parent.parent
    # ייבוא sys כדי להוסיף את src לנתיב
    import sys
    # מוסיפים את src לנתיב הייבוא
    sys.path.insert(0, str(root / "src"))
    # מייבאים את פונקציית טעינת השחקנים
    from data_loader import load_players_data

    # טוענים את שחקני FC24 בלבד
    raw = load_players_data(root / "data/raw/male_players.csv")  # FC24 only
    # מנקים אותם
    clean = clean_players_data(raw)

    # נתיב הפלט
    out = root / "data/processed/clean_players.csv"
    # שומרים ל-CSV
    clean.to_csv(out, index=False)

    # ---- quality checks ----
    # ממדי הטבלה ושם הקובץ
    print(f"clean_players: {clean.shape[0]:,} rows x {clean.shape[1]} cols -> {out.name}")
    # מאמתים שאין מזהי שחקן כפולים
    print("duplicate player_id:", clean["player_id"].duplicated().sum())
    # כמה ערכי NaN בדירוג הכללי
    print("NaN in overall:", clean["overall"].isna().sum())
    # כמה ערכי NaN בשווי
    print("NaN in value_eur:", clean["value_eur"].isna().sum())
    # פילוח לפי קבוצת עמדה
    print("\nposition_group counts:")
    print(clean["position_group"].value_counts(dropna=False).to_string())
    # ממוצע ציון היכולת לכל קבוצת עמדה
    print("\nability_score by group (mean):")
    print(clean.groupby("position_group")["ability_score"].mean().round(2).to_string())
    # 5 המציאות המובילות לפי ציון יעילות שוק
    print("\nTop 5 by market_efficiency_score (candidate bargains):")
    # עמודות לתצוגה
    cols = ["short_name", "position_group", "overall", "ability_score",
            "value_eur", "market_efficiency_score"]
    # מדפיסים את 5 הגדולים בציון היעילות
    print(clean.nlargest(5, "market_efficiency_score")[cols].to_string(index=False))


# נקודת כניסה
if __name__ == "__main__":
    main()
