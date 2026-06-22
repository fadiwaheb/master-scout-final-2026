"""
player_event_stats.py — Stage 6.
Aggregate player_match_stats to ONE row per player, with totals, per-match
values, rates (0-1), four computed scores, and matches_with_2_plus_goals.

Output: data/processed/player_event_stats.csv  (one row per player)

THRESHOLDS / WEIGHTS (documented in docs/ספים_והחלטות_Master_Scout.docx):
  MIN_MATCHES        — minimum matches for reliable rates/scores
  MIN_FOOT_SHOTS     — minimum foot shots for a meaningful foot-balance score
  DISCIPLINE_CARD_CAP— weighted cards/match that maps to discipline_score = 0
  ATTACKING_WEIGHTS / CREATIVE_WEIGHTS — weights inside the raw attacking/creative blends
"""

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא numpy לחישובים מספריים ולערכי NaN
import numpy as np
# ייבוא pandas לעבודה עם הטבלאות
import pandas as pd

# =====================================================================
# THRESHOLDS & WEIGHTS
# =====================================================================

# Minimum matches to trust a player's rates/scores.
# Reasoning: rates from 1-2 matches are noise (a 1-match player with 1 goal
# would show goals_per_match=1.0). Below this, the percentile scores are NaN.
# מינימום משחקים כדי לסמוך על השיעורים/הציונים של שחקן (פחות מזה = רעש)
MIN_MATCHES = 5

# Minimum foot shots (left+right) for a foot-balance score.
# Reasoning: you cannot judge two-footedness from 1-2 shots.
# מינימום בעיטות רגל (שמאל+ימין) כדי לחשב ציון איזון דו-רגלי אמין
MIN_FOOT_SHOTS = 5

# Weighted cards per match that maps discipline_score to 0 (worst).
# Reasoning: a player averaging >= 0.5 weighted cards per match is highly
# undisciplined; 0 cards -> score 100. (yellow=1, red=3 weight.)
# סף כרטיסים משוקללים למשחק שממפה את ציון המשמעת ל-0 (הגרוע ביותר)
DISCIPLINE_CARD_CAP = 0.5
# משקל כרטיס אדום ביחס לצהוב (אדום שווה 3 צהובים)
RED_CARD_WEIGHT = 3

# Raw attacking blend (per-match), goals weighted highest. Turned into a
# 0-100 percentile across qualified players.
# משקלי התערובת ההתקפית (לפי משחק) — לגולים המשקל הגבוה ביותר
ATTACKING_WEIGHTS = {
    # גולים למשחק — המשקל הדומיננטי
    "goals_per_match": 3.0,
    # מסירות מפתח למשחק
    "key_passes_per_match": 2.0,
    # בעיטות מהרחבה למשחק
    "box_shots_per_match": 1.0,
    # בעיטות למשחק
    "shots_per_match": 0.5,
}
# Raw creative blend (per-match); through balls are rarer/more creative.
# משקלי תערובת היצירתיות (לפי משחק) — מסירות עומק נדירות ויצירתיות יותר
CREATIVE_WEIGHTS = {
    # בישולי מסירת עומק למשחק — המשקל הגבוה
    "through_ball_assists_per_match": 3.0,
    # מסירות מפתח למשחק
    "key_passes_per_match": 1.0,
}

# columns in player_match_stats to sum into season totals
# מיפוי עמודות שחקן×משחק לסיכומים העונתיים שלהן
TOTAL_MAP = {
    # סך בעיטות
    "shots": "total_shots",
    # סך גולים
    "goals": "total_goals",
    # סך מסירות מפתח
    "key_passes": "total_key_passes",
    # סך בעיטות מהרחבה
    "box_shots": "total_box_shots",
    # סך בעיטות למסגרת
    "shots_on_target": "total_shots_on_target",
    # סך בעיטות רגל שמאל
    "left_foot_shots": "total_left_foot_shots",
    # סך בעיטות רגל ימין
    "right_foot_shots": "total_right_foot_shots",
    # סך נגיחות
    "header_shots": "total_header_shots",
    # סך כרטיסים צהובים
    "yellow_cards": "total_yellow_cards",
    # סך כרטיסים אדומים
    "red_cards": "total_red_cards",
    # סך עבירות
    "fouls": "total_fouls",
    # סך בישולי מסירת עומק
    "through_ball_assists": "total_through_ball_assists",
}


# פונקציית עזר: חלוקה איבר-איבר שמחזירה NaN במקום חלוקה באפס
def _safe_div(a, b):
    """Element-wise a/b, returning NaN where b == 0."""
    # מחליפים 0 ב-NaN כדי למנוע חלוקה באפס
    b = b.replace(0, np.nan)
    # מחזירים את המנה
    return a / b


# ---- the four computed scores ----

# ציון איזון דו-רגלי 0–100 לפי יחס בעיטות הרגל החלשה לחזקה
def calculate_foot_balance_score(df):
    """0-100 two-footedness from shot feet. 100 = perfectly balanced.
    NaN if foot shots < MIN_FOOT_SHOTS."""
    # בעיטות רגל שמאל
    left = df["total_left_foot_shots"]
    # בעיטות רגל ימין
    right = df["total_right_foot_shots"]
    # סך בעיטות הרגליים
    foot_total = left + right
    # הרגל החלשה (המינימום בין שתיהן)
    weaker = np.minimum(left, right)
    # הרגל החזקה (המקסימום), עם החלפת 0 ב-NaN למניעת חלוקה באפס
    stronger = np.maximum(left, right).replace(0, np.nan)
    # הציון: יחס חלשה/חזקה כפול 100 (0 = רגל אחת, 100 = מאוזן)
    score = (weaker / stronger) * 100      # 0 = one-footed, 100 = even
    # מי שמתחת לסף בעיטות הרגל מקבל NaN
    score[foot_total < MIN_FOOT_SHOTS] = np.nan
    # מעגלים ומחזירים
    return score.round(2)


# ציון משמעת 0–100 לפי כרטיסים למשחק (100 = אף כרטיס)
def calculate_discipline_score(df):
    """0-100 discipline. 100 = never booked; 0 = >= DISCIPLINE_CARD_CAP
    weighted cards per match."""
    # כרטיסים משוקללים: צהוב=1, אדום=3
    weighted_cards = df["total_yellow_cards"] + RED_CARD_WEIGHT * df["total_red_cards"]
    # כרטיסים משוקללים למשחק
    cards_per_match = weighted_cards / df["matches"]
    # הציון: 100 כשאין כרטיסים, יורד עד 0 בסף הכרטיסים
    score = 100 * (1 - np.minimum(cards_per_match / DISCIPLINE_CARD_CAP, 1.0))
    # מי שמתחת למינימום המשחקים מקבל NaN
    score[df["matches"] < MIN_MATCHES] = np.nan
    # מעגלים ומחזירים
    return score.round(2)


# פונקציית עזר: בונה תערובת משוקללת והופכת אותה לאחוזון 0–100 בין השחקנים הכשירים
def _weighted_percentile_score(df, weights):
    """Build a raw weighted blend, then convert to a 0-100 percentile across
    players with matches >= MIN_MATCHES. Others get NaN."""
    # מי שיש לו מספיק משחקים נחשב כשיר
    qualified = df["matches"] >= MIN_MATCHES
    # התערובת הגולמית: סכום משוקלל של העמודות
    raw = sum(w * df[col] for col, w in weights.items())
    # משאירים ערך רק לכשירים (אחרת NaN)
    raw = raw.where(qualified)
    # ממירים לאחוזון 0–100
    pct = raw.rank(pct=True) * 100
    # מעגלים ומחזירים
    return pct.round(2)


# ציון מעורבות התקפית — אחוזון של התערובת ההתקפית (כבדת-גולים)
def calculate_attacking_involvement_score(df):
    """0-100 percentile of a per-match attacking blend (goals-heavy)."""
    # מחזירים את האחוזון לפי משקלי ההתקפה
    return _weighted_percentile_score(df, ATTACKING_WEIGHTS)


# ציון יצירתיות — אחוזון של תערובת היצירתיות (מסירות מפתח/עומק)
def calculate_creative_score(df):
    """0-100 percentile of a per-match creativity blend (key passes / through balls)."""
    # מחזירים את האחוזון לפי משקלי היצירתיות
    return _weighted_percentile_score(df, CREATIVE_WEIGHTS)


# פונקציה שמכווצת את טבלת שחקן×משחק לשורה אחת לכל שחקן
def build_player_event_stats(pms):
    """Aggregate player_match_stats (player x match) to one row per player."""
    # מקבצים לפי שם השחקן
    g = pms.groupby("clean_name")

    # סוכמים כל עמודה לסיכום העונתי שלה
    out = g.agg(**{name: (src, "sum") for src, name in TOTAL_MAP.items()})
    # מספר המשחקים = מספר מזהי משחק ייחודיים
    out["matches"] = g["id_odsp"].nunique()
    # מספר משחקי הדאבל (2+ גולים)
    out["matches_with_2_plus_goals"] = g["goals"].apply(lambda s: int((s >= 2).sum()))
    # מאפסים אינדקס כדי ש-clean_name יהיה עמודה
    out = out.reset_index()

    # ---- per-match columns ----
    # גולים למשחק
    out["goals_per_match"] = _safe_div(out["total_goals"], out["matches"])
    # בעיטות למשחק
    out["shots_per_match"] = _safe_div(out["total_shots"], out["matches"])
    # מסירות מפתח למשחק
    out["key_passes_per_match"] = _safe_div(out["total_key_passes"], out["matches"])
    # בעיטות מהרחבה למשחק
    out["box_shots_per_match"] = _safe_div(out["total_box_shots"], out["matches"])
    # בישולי מסירת עומק למשחק
    out["through_ball_assists_per_match"] = _safe_div(
        out["total_through_ball_assists"], out["matches"])

    # ---- rates (0-1) ----
    # דיוק בעיטות = בעיטות למסגרת חלקי סך בעיטות
    out["shot_accuracy"] = _safe_div(out["total_shots_on_target"], out["total_shots"])
    # אחוז המרה = גולים חלקי בעיטות
    out["conversion_rate"] = _safe_div(out["total_goals"], out["total_shots"])
    # אחוז בעיטות מהרחבה מתוך כלל הבעיטות
    out["box_shot_rate"] = _safe_div(out["total_box_shots"], out["total_shots"])

    # ---- four computed scores ----
    # ציון איזון דו-רגלי
    out["foot_balance_score"] = calculate_foot_balance_score(out)
    # ציון משמעת
    out["discipline_score"] = calculate_discipline_score(out)
    # ציון מעורבות התקפית
    out["attacking_involvement_score"] = calculate_attacking_involvement_score(out)
    # ציון יצירתיות
    out["creative_score"] = calculate_creative_score(out)

    # מחזירים את הטבלה ברמת שחקן
    return out


# בלוק שמורץ בהרצה ישירה — בונה ומאמת את טבלת ציוני האירועים
def main():
    # שורש הפרויקט
    root = Path(__file__).resolve().parent.parent
    # טוענים את טבלת שחקן×משחק
    pms = pd.read_csv(root / "data/processed/player_match_stats.csv")
    # בונים את טבלת ציוני האירועים לרמת שחקן
    pes = build_player_event_stats(pms)

    # נתיב הפלט
    out = root / "data/processed/player_event_stats.csv"
    # שמירה ל-CSV
    pes.to_csv(out, index=False)

    # ---- quality checks ----
    # ממדי הטבלה ושם הקובץ
    print(f"player_event_stats: {pes.shape[0]:,} rows x {pes.shape[1]} cols -> {out.name}")
    # אימות שגולים-למשחק = גולים חלקי משחקים
    print("reconcile goals_per_match = total_goals/matches:",
          bool(np.allclose((pes["total_goals"] / pes["matches"]),
                           pes["goals_per_match"], equal_nan=True)))
    # בודקים שכל השיעורים בטווח 0..1
    for c in ["shot_accuracy", "conversion_rate", "box_shot_rate"]:
        # מסירים NaN לפני בדיקת הטווח
        v = pes[c].dropna()
        # מדפיסים מינימום ומקסימום
        print(f"  {c}: min={v.min():.3f} max={v.max():.3f}  (expect 0..1)")
    # סופרים אם יש שיעור כלשהו מעל 1 (מצופה 0)
    print("any rate > 1:", int(((pes[["shot_accuracy", "conversion_rate", "box_shot_rate"]] > 1)
                                 .sum().sum())))
    # כמה שחקנים עומדים במינימום המשחקים
    print("players with matches >= MIN_MATCHES:", int((pes["matches"] >= MIN_MATCHES).sum()))
    # 5 המובילים במעורבות התקפית
    print("\nTop 5 attacking_involvement_score:")
    # עמודות לתצוגה
    cols = ["clean_name", "matches", "total_goals", "goals_per_match",
            "attacking_involvement_score", "creative_score"]
    # מדפיסים אותם
    print(pes.nlargest(5, "attacking_involvement_score")[cols].to_string(index=False))
    # 5 המובילים ביצירתיות
    print("\nTop 5 creative_score:")
    print(pes.nlargest(5, "creative_score")[cols].to_string(index=False))
    # 5 השחקנים הדו-רגליים ביותר (עם סף בעיטות רגל)
    print("\nMost two-footed (foot_balance_score, min foot shots applied):")
    # מסירים NaN בציון האיזון
    fb = pes.dropna(subset=["foot_balance_score"])
    # מדפיסים את 5 המובילים
    print(fb.nlargest(5, "foot_balance_score")[
        ["clean_name", "total_left_foot_shots", "total_right_foot_shots",
         "foot_balance_score"]].to_string(index=False))


# נקודת כניסה
if __name__ == "__main__":
    main()
