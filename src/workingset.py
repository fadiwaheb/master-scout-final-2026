"""
workingset.py — the "funnel" engine.

Lets the chat narrow the big table to a smaller WORKING SET, query after query.
Example: "forwards 25-30"  ->  then "of those, who has 15+ braces"  ->  then ...
Each step returns the FULL-column subset of the players that match so far, so the
next query can filter on any column. "start over" resets to the full table.

The chat passes the current working set as the source to the agent, and after a
successful query calls narrow() to compute the next working set.
"""

# מאפשר תחביר טיפוסים מודרני (pd.DataFrame | None)
from __future__ import annotations

# ייבוא pandas לעבודה עם הטבלאות
import pandas as pd

# intents that narrow by FILTERING rows (every matching player stays in the set)
# כוונות שמצמצמות על-ידי סינון שורות (כל שחקן שעומד בתנאי נשאר בקבוצה)
_FILTER_NARROW = {
    "profile_search", "attacking_players", "creative_midfielders",
    "disciplined_defenders", "two_footed", "braces",
}
# intents that narrow to the MODEL-RANKED players that were shown
# כוונות שמצמצמות לשחקנים שדורגו על-ידי המודל והוצגו (דמיון/מציאות/חריגות)
_RESULT_NARROW = {"similar_players", "bargains", "profile_performance_anomaly"}

# implicit position for the position-specific performance intents
# עמדה מובלעת עבור כוונות ביצועים תלויות-עמדה (יצירתיות=קשר, משמעת=הגנה)
_IMPLICIT_POS = {"creative_midfielders": "Midfielder",
                 "disciplined_defenders": "Defender"}

# curated "scout view" columns (only those present are shown) + Hebrew legend
# עמודות "מבט הסקאוט" המוצגות (רק אלה שקיימות) — תצוגה נקייה למשתמש
SCOUT_COLS = [
    # זהות ופרופיל בסיסי
    "short_name", "position_group", "age", "overall", "potential", "value_eur",
    # מועדון/ליגה/נבחרת ורגל
    "preferred_foot", "club_name", "league_name", "nationality_name",
    # נתוני ביצועים: משחקים, גולים, גולים למשחק, דאבלים
    "matches", "total_goals", "goals_per_match", "matches_with_2_plus_goals",
    # ציונים מחושבים
    "attacking_involvement_score", "creative_score", "discipline_score",
    "foot_balance_score",
]
# מקרא עברי: תרגום שם כל עמודה להסבר קצר (לתצוגה מתחת לטבלה)
LEGEND = {
    "short_name": "שם השחקן", "position_group": "עמדה", "age": "גיל",
    "overall": "דירוג כללי (FC24)", "potential": "פוטנציאל", "value_eur": "שווי שוק (€)",
    "preferred_foot": "רגל חזקה", "club_name": "מועדון", "league_name": "ליגה",
    "nationality_name": "נבחרת", "matches": "משחקים (אירועים)", "total_goals": "גולים",
    "goals_per_match": "גולים למשחק", "matches_with_2_plus_goals": "משחקי דאבל (2+ גולים)",
    "attacking_involvement_score": "מעורבות התקפית", "creative_score": "יצירתיות",
    "discipline_score": "משמעת", "foot_balance_score": "איזון דו-רגלי",
}


# פונקציית עזר: מחילה כל מסנן שורות מוכר שנמצא ב-f ומחזירה את תת-הקבוצה המלאה
def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Apply every recognized ROW filter present in `f`; return the full subset."""
    # מתחילים מהטבלה כולה
    out = df
    # רשימת מסנני טווח: (עמודה, מפתח-מסנן, ">=" או "<=")
    rng = [  # (column, filter_key, ">=" or "<=")
        # גיל מינימלי/מקסימלי
        ("age", "min_age", "ge"), ("age", "max_age", "le"),
        # דירוג/פוטנציאל מינימליים
        ("overall", "min_overall", "ge"), ("potential", "min_potential", "ge"),
        # מהירות מינימלית
        ("pace", "min_pace", "ge"),
        # שווי מינימלי/מקסימלי
        ("value_eur", "min_value_eur", "ge"), ("value_eur", "max_value_eur", "le"),
        # real match performance
        # גולים כוללים מינימליים
        ("total_goals", "min_total_goals", "ge"),
        # מספר דאבלים מינימלי
        ("matches_with_2_plus_goals", "min_braces", "ge"),
        # גולים למשחק מינימלי
        ("goals_per_match", "min_goals_per_match", "ge"),
        # מסירות מפתח (≈ בישולים)
        ("total_key_passes", "min_key_passes", "ge"),       # ≈ assists / chance creation
        # בעיטות כוללות מינימליות
        ("total_shots", "min_shots", "ge"),
        # מקסימום כרטיסים צהובים
        ("total_yellow_cards", "max_yellow_cards", "le"),
        # מקסימום כרטיסים אדומים
        ("total_red_cards", "max_red_cards", "le"),
        # ציון מעורבות התקפית מינימלי
        ("attacking_involvement_score", "min_attacking_involvement", "ge"),
        # ציון יצירתיות מינימלי
        ("creative_score", "min_creative", "ge"),
        # ציון משמעת מינימלי
        ("discipline_score", "min_discipline", "ge"),
        # ציון איזון דו-רגלי מינימלי
        ("foot_balance_score", "min_foot_balance", "ge"),
    ]
    # סינון לפי קבוצת עמדה אם צוינה והעמודה קיימת
    if f.get("position_group") and "position_group" in out:
        out = out[out["position_group"] == f["position_group"]]
    # סינון לפי רגל מועדפת
    if f.get("preferred_foot") and "preferred_foot" in out:
        out = out[out["preferred_foot"] == f["preferred_foot"]]
    # סינון לפי ליגה (התאמה חלקית, ללא תלות באותיות)
    if f.get("league_name") and "league_name" in out:
        out = out[out["league_name"].str.contains(f["league_name"], case=False, na=False)]
    # סינון לפי נבחרת/לאום (התאמה חלקית)
    if f.get("nationality") and "nationality_name" in out:
        out = out[out["nationality_name"].str.contains(f["nationality"], case=False, na=False)]
    # מחילים את כל מסנני הטווח לפי הרשימה
    for col, key, op in rng:
        # הערך המבוקש מהמסנן
        val = f.get(key)
        # מחילים רק אם הערך קיים והעמודה קיימת
        if val is not None and col in out.columns:
            # ">=" עבור מינימום, "<=" עבור מקסימום
            out = out[out[col] >= val] if op == "ge" else out[out[col] <= val]
    # מחזירים את תת-הקבוצה המסוננת (עם כל העמודות)
    return out


# performance filters require event data; each also implies a sensible sort metric
# מסנני ביצועים שדורשים נתוני אירועים (ולכן גם מגבילים לשחקנים עם נתוני מגרש)
_PERF_KEYS = {"min_total_goals", "min_braces", "min_goals_per_match", "min_key_passes",
              "min_shots", "max_yellow_cards", "max_red_cards",
              "min_attacking_involvement", "min_creative", "min_discipline",
              "min_foot_balance"}
# מיון ברירת מחדל לכל כוונה (לפי המדד הרלוונטי לה)
_SORT_DEFAULT = {
    "profile_search": "overall", "attacking_players": "attacking_involvement_score",
    "creative_midfielders": "creative_score", "disciplined_defenders": "discipline_score",
    "two_footed": "foot_balance_score", "braces": "matches_with_2_plus_goals",
}
# מיון מועדף לפי מסנן ספציפי (אם הופעל)
_FILTER_SORT = {
    "min_total_goals": "total_goals", "min_key_passes": "total_key_passes",
    "min_shots": "total_shots", "min_braces": "matches_with_2_plus_goals",
    "min_goals_per_match": "goals_per_match",
}
# עמודות הבסיס שתמיד מוצגות בתוצאה
DISPLAY_BASE = ["short_name", "position_group", "age", "overall", "potential", "value_eur"]


# פונקציית עזר: מוסיפה אילוצים מובלעים (נתוני אירועים / עמדה מובלעת)
def _prepare(df: pd.DataFrame, intent: str, f: dict):
    """Add implicit constraints (event data for performance intents/filters, the
    implicit position for creative/disciplined). Returns (base_df, filters)."""
    # מתחילים מהטבלה הנתונה
    base = df
    # האם נדרשים נתוני אירועים (כוונת ביצועים או מסנן ביצועים כלשהו)
    needs_events = (intent in (_FILTER_NARROW - {"profile_search"})
                    or any(k in f for k in _PERF_KEYS))
    # אם כן — מגבילים לשחקנים שיש להם נתוני אירועים
    if needs_events and "has_event_data" in df.columns:
        base = df[df["has_event_data"]]
    # אם הכוונה דורשת עמדה מובלעת ולא צוינה עמדה — מוסיפים אותה
    if intent in _IMPLICIT_POS and "position_group" not in f:
        f = {**f, "position_group": _IMPLICIT_POS[intent]}
    # מחזירים את הבסיס והמסננים המעודכנים
    return base, f


# פונקציה גנרית: סינון + מיון + חיתוך ל-top_n. בסיס לחוסן מול טקסט חופשי
def search(df: pd.DataFrame, intent: str, filters: dict, top_n: int = 10):
    """Generic filter + sort + top_n. Returns (display_df, full_narrowed_df).
    Any recognized filter works here — this is what makes free text robust."""
    # מכינים בסיס + מסננים (עם אילוצים מובלעים)
    base, f = _prepare(df, intent, dict(filters or {}))
    # מחילים את כל המסננים
    sub = _apply_filters(base, f)
    # בוחרים עמודת מיון: לפי מסנן ספציפי אם הופעל, אחרת ברירת המחדל של הכוונה
    sort_by = next((_FILTER_SORT[k] for k in _FILTER_SORT if k in f),
                   _SORT_DEFAULT.get(intent, "overall"))
    # אם עמודת המיון אינה קיימת — נופלים חזרה לדירוג הכללי
    if sort_by not in sub.columns:
        sort_by = "overall"
    # ממיינים מהגבוה לנמוך
    sub = sub.sort_values(sort_by, ascending=False)
    # עמודות התצוגה מתחילות מעמודות הבסיס
    cols = list(DISPLAY_BASE)
    # מוסיפים את עמודת המיון אם אינה כלולה
    if sort_by not in cols:
        cols.append(sort_by)
    # שומרים רק עמודות שקיימות בפועל
    cols = [c for c in cols if c in sub.columns]
    # מחזירים את ה-top_n לתצוגה ואת תת-הקבוצה המלאה להמשך צמצום
    return sub.head(top_n)[cols].reset_index(drop=True), sub


# פונקציה שמחשבת את קבוצת העבודה הבאה (או None אם לא משתנה)
def narrow(source: pd.DataFrame, intent: str, filters: dict,
           display_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Return the next working set (full columns), or None to leave it unchanged."""
    # עותק של המסננים
    f = dict(filters or {})
    # כוונות סינון: קבוצת העבודה הבאה היא כל מי שעומד בתנאים
    if intent in _FILTER_NARROW:
        # מכינים בסיס + מסננים
        base, f = _prepare(source, intent, f)
        # מחזירים את תת-הקבוצה המסוננת
        return _apply_filters(base, f)

    # כוונות מדורגות-מודל: מצמצמים לשחקנים שהוצגו בפועל (לפי שם)
    if (intent in _RESULT_NARROW and display_df is not None
            and "short_name" in display_df.columns and "short_name" in source.columns):
        # שומרים מהמקור רק את השחקנים שהופיעו בתוצאה
        return source[source["short_name"].isin(display_df["short_name"])]

    # קיבוץ / ויזואליזציה / ברכה — קבוצת העבודה לא משתנה
    return None  # clustering / visualize / greeting -> working set unchanged


# פונקציה שמחזירה תת-קבוצת עמודות נקייה לתצוגה (מבט הסקאוט)
def scout_view(df: pd.DataFrame) -> pd.DataFrame:
    """Curated, readable column subset of a working set for display."""
    # שומרים רק את עמודות הסקאוט שקיימות בטבלה
    cols = [c for c in SCOUT_COLS if c in df.columns]
    # מחזירים את התצוגה
    return df[cols]
