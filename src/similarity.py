"""
similarity.py — Stage 9.  USE #1 (similarity).
Find similar players with feature normalization + Cosine similarity.

This is the numeric analog of the TF-IDF + Cosine technique from the course:
instead of TF-IDF term weights we use MinMax-normalized attribute vectors, then
measure the cosine of the angle between players.

Key deliverable for rubric ch. 5+6: a MODEL COMPARISON — run the same query
WITH the position_group category (restrict candidates to the same position) and
WITHOUT it, and document how the results differ.

Output: similarity.py (module). Similarity score is in [0, 1] because MinMax
features are non-negative.
"""

# ייבוא re לביטויים רגולריים (התאמת שמות)
import re
# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא numpy לחישובים וקטוריים
import numpy as np
# ייבוא pandas לעבודה עם הטבלה
import pandas as pd
# ייבוא StandardScaler לנרמול z-score של התכונות
from sklearn.preprocessing import StandardScaler
# ייבוא פונקציית דמיון הקוסינוס
from sklearn.metrics.pairwise import cosine_similarity

# Attribute features used for play-style similarity (FC24, available for all
# outfield players). Chosen to span shooting, passing, dribbling, defending,
# physical and movement — a rounded play-style fingerprint.
# 14 התכונות המרכיבות את "טביעת האצבע" של סגנון המשחק לחישוב הדמיון
SIM_FEATURES = [
    # 6 תכונות הליבה
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    # מיומנות וראיית משחק
    "skill_ball_control", "skill_dribbling", "mentality_vision",
    # גמר התקפי וכוח
    "attacking_finishing", "power_strength", "movement_sprint_speed",
    # תאוצה והגנה בעמידה
    "movement_acceleration", "defending_standing_tackle",
]


# the 6 "face" attributes — the only play-style features a goalkeeper lacks and
# the only ones we can get from an external source for a player outside our data.
# 6 תכונות "הפנים" — היחידות שחסרות לשוער והיחידות שניתן להביא ממקור חיצוני
FACE6 = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]


# פונקציה: כל השחקנים השונים ששמם מתאים (לצורך הבחנה בין כפילי-שם)
def find_player_matches(df, player_name, limit=8):
    """All distinct players whose name matches (for disambiguation)."""
    # מנקים את שאילתת השם (אותיות קטנות, ללא רווחים בקצוות)
    q = player_name.strip().lower()
    # מורידים לאותיות קטנות את השם הקצר והשם המלא
    sn, ln = df["short_name"].str.lower(), df["long_name"].str.lower()
    # מסכת התאמה: השאילתה מופיעה בשם הקצר או המלא
    mask = sn.str.contains(re.escape(q), na=False) | ln.str.contains(re.escape(q), na=False)
    # השחקנים התואמים
    hits = df[mask]
    # אם אין התאמה — ננסה התאמת כל המילים (token-AND)
    if hits.empty:  # token-AND fallback
        # מפרקים את השאילתה למילים
        toks = [t for t in re.split(r"\s+", q) if t]
        # אם יש מילים
        if toks:
            # מסכה שמתחילה מ-True לכל השורות
            m = pd.Series(True, index=df.index)
            # כל מילה חייבת להופיע בשם
            for t in toks:
                m &= (sn.str.contains(re.escape(t), na=False) | ln.str.contains(re.escape(t), na=False))
            # ההתאמות החדשות
            hits = df[m]
    # מחזירים את ההתאמות ממוינות לפי דירוג, עד למגבלה
    return hits.sort_values("overall", ascending=False).head(limit)


# פונקציה: דמיון קוסינוס מול המאגר לפי 6 תכונות הפנים — לשחקן יעד שאינו במאגר
def find_similar_to_attrs(df, attrs, top_n=10, position_group=None):
    """Cosine similarity against the pool using ONLY the 6 face attributes — used
    for a target player that is NOT in our data (attributes come from an external
    source). Returns a DataFrame with a `similarity` column."""
    # מאגר המועמדים: שחקני שדה בלבד עם כל 6 תכונות הפנים
    pool = df[df["position_group"] != "GK"].dropna(subset=FACE6).copy()
    # אם צוינה עמדה — מצמצמים אליה
    if position_group:
        pool = pool[pool["position_group"] == position_group]
    # וקטור היעד: 6 התכונות מהמקור החיצוני
    target = np.array([[float(attrs[a]) for a in FACE6]])
    # מאחדים את המאגר עם וקטור היעד לצורך נרמול משותף
    X = np.vstack([pool[FACE6].astype(float).values, target])
    # נרמול z-score של כל התכונות
    Z = StandardScaler().fit_transform(X)
    # דמיון קוסינוס בין היעד (השורה האחרונה) לכל המאגר
    cos = cosine_similarity(Z[-1:], Z[:-1])[0]
    # ממפים את הדמיון מ-[-1,1] ל-[0,1] ומעגלים, ומוסיפים כעמודה
    pool = pool.assign(similarity=np.round((cos + 1) / 2, 4))
    # ממיינים לפי דמיון ולוקחים את ה-top_n
    pool = pool.sort_values("similarity", ascending=False).head(top_n)
    # עמודות התצוגה
    cols = ["short_name", "position_group", "age", "overall", "value_eur", "similarity"]
    # מחזירים את התוצאה
    return pool[cols].reset_index(drop=True)


# פונקציית עזר: איתור שורת שחקן לפי שם חופשי (שם קצר/מלא/חלקי)
def _find_player_row(df, player_name):
    """Locate a player by a free-text name (matches short_name or long_name,
    case-insensitive). Works for short names, full names and partials —
    e.g. "Messi", "Lionel Messi" and "L. Messi" all resolve to the same row.
    If several match, return the highest-overall one."""
    # מנקים את השאילתה
    q = player_name.strip().lower()
    # שם קצר באותיות קטנות
    sn = df["short_name"].str.lower()
    # שם מלא באותיות קטנות
    ln = df["long_name"].str.lower()

    # 1) direct substring match (handles "messi", "l. messi")
    # התאמה ישירה של תת-מחרוזת בשם הקצר או המלא
    mask = sn.str.contains(re.escape(q), na=False) | ln.str.contains(re.escape(q), na=False)
    # ההתאמות
    hits = df[mask]

    # 2) fall back to token-AND: every word in the query appears in the name
    #    (handles "lionel messi" vs long_name "Lionel Andrés Messi Cuccittini")
    # אם אין התאמה — נדרוש שכל מילה בשאילתה תופיע בשם
    if hits.empty:
        # מפרקים למילים
        tokens = [t for t in re.split(r"\s+", q) if t]
        # אם יש מילים
        if tokens:
            # מסכה שמתחילה מ-True
            tok_mask = pd.Series(True, index=df.index)
            # כל מילה חייבת להימצא
            for t in tokens:
                tok_mask &= (sn.str.contains(re.escape(t), na=False)
                             | ln.str.contains(re.escape(t), na=False))
            # ההתאמות לפי כל המילים
            hits = df[tok_mask]

    # אם אין התאמה כלל — מחזירים None
    if hits.empty:
        return None
    # מחזירים את ההתאמה בעלת הדירוג הגבוה ביותר
    return hits.sort_values("overall", ascending=False).iloc[0]


# פונקציה ראשית: מחזירה את top_n השחקנים הדומים ביותר לשחקן נתון
def find_similar_players(df, player_name, top_n=10, same_position=True,
                         max_age=None, max_value_eur=None):
    """Return the top_n players most similar to `player_name`.

    same_position=True  -> WITH category: candidates limited to the target's
                           position_group (the course's "with category" run).
    same_position=False -> WITHOUT category: candidates are all outfield players.

    Returns a DataFrame with a `similarity` column in [0, 1] and a short
    `reason` naming the closest shared attributes. The target is never returned.
    """
    # מאתרים את שחקן היעד לפי שמו
    target = _find_player_row(df, player_name)
    # אם לא נמצא — שגיאה
    if target is None:
        raise ValueError(f"player not found: {player_name!r}")

    # goalkeepers lack the 6 outfield play-style attributes, so a play-style
    # vector for them is empty — Cosine similarity can't be computed. Signal it
    # clearly (callers turn this into a friendly "outfield players only" notice)
    # instead of returning meaningless NaN matches.
    # שוער חסר את תכונות שחקן השדה — אי אפשר לחשב דמיון, ולכן נסמן זאת בבירור
    if str(target["position_group"]) == "GK":
        raise ValueError("goalkeeper play-style analysis not supported")

    # candidate pool: outfield players with complete features
    # מאגר המועמדים: שחקני שדה בלבד עם כל תכונות הדמיון
    pool = df[df["position_group"] != "GK"].dropna(subset=SIM_FEATURES).copy()
    # WITH category — מצמצמים לאותה עמדה של היעד
    if same_position:
        pool = pool[pool["position_group"] == target["position_group"]]
    # מסנן גיל מקסימלי אם צוין
    if max_age is not None:
        pool = pool[pool["age"] <= max_age]
    # מסנן שווי מקסימלי אם צוין
    if max_value_eur is not None:
        pool = pool[pool["value_eur"] <= max_value_eur]

    # make sure the target is part of the scaling space
    # מוודאים שהיעד נמצא במאגר כדי שייכלל במרחב הנרמול
    if target["player_id"] not in set(pool["player_id"]):
        pool = pd.concat([pool, target.to_frame().T], ignore_index=True)

    # z-score normalize each attribute (StandardScaler) so cosine compares the
    # ABOVE/BELOW-average play-style pattern — far more discriminative than raw
    # positive vectors. Cosine in [-1,1] is then rescaled to [0,1] via (x+1)/2,
    # so 1 = identical style, 0.5 = unrelated, 0 = opposite style.
    # נרמול z-score של התכונות — כדי שהקוסינוס ישווה דפוס מעל/מתחת לממוצע
    scaler = StandardScaler()
    # מטריצת התכונות המנורמלת
    X = scaler.fit_transform(pool[SIM_FEATURES].astype(float))

    # מאתרים את אינדקס שורת היעד במאגר
    target_idx = pool.index[pool["player_id"] == target["player_id"]][0]
    # וקטור היעד המנורמל
    target_vec = X[pool.index.get_loc(target_idx)].reshape(1, -1)
    # דמיון קוסינוס בין היעד לכל המאגר
    cos = cosine_similarity(target_vec, X)[0]
    # ממפים מ-[-1,1] ל-[0,1]
    sims = (cos + 1) / 2  # map [-1,1] -> [0,1]

    # מוסיפים את ערכי הדמיון כעמודה (מעוגלים)
    pool = pool.assign(similarity=np.round(sims, 4))
    # מסירים את היעד עצמו מהתוצאות
    pool = pool[pool["player_id"] != target["player_id"]]  # drop self
    # ממיינים לפי דמיון ולוקחים את ה-top_n
    pool = pool.sort_values("similarity", ascending=False).head(top_n)

    # human-readable reason: the attributes where the two players are closest
    # נימוק קריא: התכונות שבהן השחקנים הכי קרובים
    pool["reason"] = pool.apply(
        lambda r: _reason(target, r), axis=1)

    # עמודות התצוגה
    cols = ["short_name", "position_group", "age", "overall", "value_eur",
            "similarity", "reason"]
    # מחזירים את התוצאה ואת שורת היעד
    return pool[cols].reset_index(drop=True), target


# פונקציית עזר: שם k התכונות שבהן המועמד הכי קרוב ליעד (לנימוק)
def _reason(target, cand, k=3):
    """Name the k attributes where candidate is closest to the target."""
    # מחשבים את ההפרש המוחלט בכל אחת מ-6 תכונות הליבה
    diffs = {f: abs(float(target[f]) - float(cand[f])) for f in
             ["pace", "shooting", "passing", "dribbling", "defending", "physic"]}
    # בוחרים את k התכונות עם ההפרש הקטן ביותר
    closest = sorted(diffs, key=diffs.get)[:k]
    # מחזירים מחרוזת נימוק
    return "similar " + ", ".join(closest)


# פונקציה: השוואת מודלים — אותה שאילתה עם ובלי קטגוריית העמדה (לדרישת המחוון)
def compare_with_without_category(df, player_name, top_n=10):
    """MODEL COMPARISON for the rubric: same query, with vs without the
    position_group category. Returns (with_cat, without_cat, overlap_count)."""
    # ריצה עם קטגוריה (אותה עמדה)
    with_cat, _ = find_similar_players(df, player_name, top_n=top_n, same_position=True)
    # ריצה בלי קטגוריה (כל שחקני השדה)
    without_cat, _ = find_similar_players(df, player_name, top_n=top_n, same_position=False)
    # החפיפה בין שתי הרשימות
    overlap = set(with_cat["short_name"]) & set(without_cat["short_name"])
    # מחזירים את שתי הרשימות ואת גודל החפיפה
    return with_cat, without_cat, len(overlap)


# בלוק שמורץ בהרצה ישירה — הדגמת הדמיון והשוואת המודלים
def main():
    # שורש הפרויקט
    root = Path(__file__).resolve().parent.parent
    # טוענים את הטבלה המרכזית
    df = pd.read_csv(root / "data/processed/final_scouting_table.csv")

    # כותרת הדגמה
    print("=" * 70, "\nDEMO — Stage 9 Cosine similarity\n", "=" * 70)
    # שחקן לדוגמה
    name = "De Bruyne"

    # מריצים דמיון עם קטגוריה
    res, target = find_similar_players(df, name, top_n=8, same_position=True)
    # מדפיסים פרטי יעד
    print(f"\nTarget: {target['short_name']} ({target['position_group']}, "
          f"overall {target['overall']})")
    # כותרת בדיקות איכות
    print("\nQuality checks:")
    # טווח ערכי הדמיון (מצופה 0..1)
    print("  similarity range:", res['similarity'].min(), "..", res['similarity'].max(),
          "(expect 0..1)")
    # מוודאים שהיעד אינו מופיע כתוצאה של עצמו
    print("  target returned as own match:",
          target['short_name'] in set(res['short_name']), "(expect False)")
    # מדפיסים את ההתאמות עם הקטגוריה
    print("\n[WITH category] most similar to", target['short_name'], ":")
    print(res.to_string(index=False))

    # כותרת השוואת מודלים
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — with vs without position_group category")
    print("=" * 70)
    # מריצים את ההשוואה
    wc, woc, overlap = compare_with_without_category(df, name, top_n=8)
    # מדפיסים את התוצאות ללא קטגוריה
    print("\n[WITHOUT category] (candidates = all outfield positions):")
    print(woc[["short_name", "position_group", "overall", "similarity", "reason"]].to_string(index=False))
    # מדפיסים את גודל החפיפה
    print(f"\nOverlap between the two top-8 lists: {overlap}/8")
    # פרשנות התוצאה
    print("Interpretation: WITHOUT the category, players from OTHER positions with "
          "a similar statistical profile can appear; WITH the category we stay "
          "within the same role — usually more actionable for scouting.")


# נקודת כניסה
if __name__ == "__main__":
    main()
