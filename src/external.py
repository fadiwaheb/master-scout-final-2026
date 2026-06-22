"""
external.py — REAL external data source: the EA Sports FC official ratings API.

The ratings site https://www.ea.com/games/ea-sports-fc/ratings is backed by a
public JSON API (drop-api.ea.com). We query it directly — no fragile HTML
scraping, no browser — to fetch live, accurate attributes (and a player photo)
for players that are NOT in our local datasets.

Source priority in the agent: our FC24 data -> our events data -> THIS (EA FC
official ratings) -> the model's own football knowledge (last resort).
"""
# מאפשר תחביר טיפוסים מודרני (כמו dict | None) גם בפייתון ישן יותר
from __future__ import annotations

# ייבוא datetime לחישוב גיל מתאריך לידה
import datetime as _dt

# מנסים לייבא את ספריית requests לקריאות HTTP; אם אין — נמשיך בלעדיה
try:
    import requests
except Exception:  # pragma: no cover
    # אם הייבוא נכשל מסמנים requests כ-None כדי שהפונקציות יחזירו None בחן
    requests = None

# כתובת ה-API הציבורי של דירוגי EA Sports FC
EA_API = "https://drop-api.ea.com/rating/ea-sports-fc"
# שם המקור לתצוגה (עברית)
SOURCE_NAME = "EA Sports FC — דירוגים רשמיים"
# כתובת אתר הדירוגים לציון מקור
SOURCE_URL = "https://www.ea.com/games/ea-sports-fc/ratings"

# מיפוי סוג העמדה של EA לאחת מ-4 קבוצות העמדה שלנו
_POS = {"attack": "Forward", "midfield": "Midfielder",
        "defense": "Defender", "goalkeeper": "GK"}
# מיפוי קוד הרגל המועדפת (1=ימין, 2=שמאל)
_FOOT = {1: "Right", 2: "Left"}


# פונקציית עזר: מחשבת גיל מתאריך לידה בפורמט של EA
def _age(birthdate: str):
    # מנסים לפרסר את התאריך ולחשב גיל
    try:
        # ממירים את מחרוזת התאריך (MM/DD/YYYY) לאובייקט תאריך
        d = _dt.datetime.strptime(birthdate.split()[0], "%m/%d/%Y").date()
        # התאריך של היום
        t = _dt.date.today()
        # מחשבים את הגיל בהתחשב אם יום ההולדת כבר עבר השנה
        return t.year - d.year - ((t.month, t.day) < (d.month, d.day))
    except Exception:
        # אם הפרסור נכשל — אין גיל
        return None


# פונקציה ראשית: מחפשת שחקן ב-API הרשמי של EA ומחזירה כרטיס בסכמה שלנו
def ea_fc_lookup(name: str, timeout: int = 8) -> dict | None:
    """Look a player up in EA Sports FC's official ratings API.
    Returns a dict in our card schema (with avatar_url + source), or None."""
    # אם אין ספריית requests או אין שם — אין מה לחפש
    if requests is None or not name:
        return None
    # מנסים לבצע את הקריאה ל-API
    try:
        # שולחים בקשת GET עם פרמטרי שפה, מגבלה וחיפוש, עם User-Agent של דפדפן
        r = requests.get(EA_API, params={"locale": "en", "limit": 1, "search": name},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        # מחלצים את רשימת הפריטים מתוך ה-JSON
        items = r.json().get("items", [])
    except Exception:
        # כל כשל ברשת/פרסור → אין תוצאה
        return None
    # אם לא נמצאו פריטים — אין שחקן
    if not items:
        return None

    # לוקחים את הפריט הראשון (ההתאמה הטובה ביותר)
    it = items[0]
    # מילון התכונות של השחקן (עשוי להיות ריק)
    s = it.get("stats", {}) or {}

    # פונקציית עזר פנימית לחילוץ ערך תכונה מתוך מבנה ה-stats
    def stat(k):
        # ערך התכונה לפי המפתח
        v = s.get(k)
        # התכונה היא מילון עם שדה value — מחזירים אותו, אחרת None
        return v.get("value") if isinstance(v, dict) else None

    # מרכיבים שם מלא מהשם הפרטי ושם המשפחה
    full = f"{it.get('firstName', '')} {it.get('lastName', '')}".strip()
    # מחלצים את מזהה סוג העמדה (attack/midfield/defense/goalkeeper)
    pos_type = ((it.get("position") or {}).get("positionType") or {}).get("id")
    # מחזירים מילון כרטיס בסכמה האחידה של המערכת
    return {
        # שם קצר: שם נפוץ, אחרת שם מלא, אחרת מה שחיפשו
        "short_name": it.get("commonName") or full or name,
        # שם מלא
        "long_name": full or it.get("commonName"),
        # קבוצת העמדה הממופית
        "position_group": _POS.get(pos_type),
        # גיל מחושב מתאריך הלידה
        "age": _age(it.get("birthdate", "")),
        # דירוג כללי
        "overall": it.get("overallRating"),
        "potential": it.get("overallRating"),       # API has no potential
        "value_eur": None,                           # API has no market value
        # רגל מועדפת ממופית
        "preferred_foot": _FOOT.get(it.get("preferredFoot")),
        # שם הלאום
        "nationality_name": (it.get("nationality") or {}).get("label"),
        # שם המועדון
        "club_name": (it.get("team") or {}).get("label"),
        # 6 תכונות הליבה מתוך ה-API
        "pace": stat("pac"), "shooting": stat("sho"), "passing": stat("pas"),
        "dribbling": stat("dri"), "defending": stat("def"), "physic": stat("phy"),
        # כתובת תמונת השחקן
        "avatar_url": it.get("avatarUrl"),
        # כתובת תמונת דגל הלאום
        "nationality_flag_url": (it.get("nationality") or {}).get("imageUrl"),
        # כתובת סמל המועדון
        "club_badge_url": (it.get("team") or {}).get("imageUrl"),
        # שחקן מ-EA — אין לו נתוני אירועים אצלנו
        "has_event_data": False,
        # מסמנים שהמקור הוא EA וכתובת המקור
        "_source": "ea", "_source_url": SOURCE_URL,
    }


# פונקציית עזר: מסירה ראשי תיבה מהשם ('L. Messi' → 'Messi') כי EA מחפש לפי שם משפחה
def _strip_initial(name: str) -> str:
    """'L. Messi' -> 'Messi' — EA search matches the surname, not the 'X.' form."""
    # ייבוא re לביטוי הרגולרי
    import re
    # מסירים תבנית של אות גדולה ונקודה בתחילת השם
    return re.sub(r"^[A-Z]\.\s*", "", str(name)).strip()


# פונקציה קלילה: מביאה רק תמונה + דגל + סמל לשחקן (להעשרת כרטיסים שכבר במאגר)
def ea_media(name: str) -> dict:
    """Lightweight: just the photo + flag + badge for a player (to enrich the cards
    of players that ARE in our data). Tries the name and its surname form."""
    # מנסים גם את השם המלא וגם את צורת שם המשפחה
    for q in (name, _strip_initial(name)):
        # מחפשים ב-EA
        d = ea_fc_lookup(q)
        # אם נמצאה תמונה — מחזירים תמונה, דגל וסמל
        if d and d.get("avatar_url"):
            return {"avatar_url": d.get("avatar_url"),
                    "flag_url": d.get("nationality_flag_url"),
                    "badge_url": d.get("club_badge_url")}
    # אם לא נמצא דבר — מילון ריק
    return {}
