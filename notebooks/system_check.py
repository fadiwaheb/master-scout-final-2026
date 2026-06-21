# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (masterscout)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <div dir="rtl" align="right">
#
# # ⚽ Master Scout — מחברת בדיקת המערכת
#
# **פרויקט גמר · סדנת AI & ML · קורס 277302**
#
# מחברת זו היא **בדיקת שפיות** לכל שלבי עיבוד הנתונים. היא טוענת את הטבלאות
# שנבנו, מציגה אותן, ומריצה את בדיקות האיכות — כדי לוודא שהכול עובד כמו שצריך.
#
# ### איך להריץ
# 1. הפעל את הסביבה: `conda activate masterscout`
# 2. פתח את המחברת ב-Jupyter/VSCode ולחץ **Run All** (חובה — כדי שכל הגרפים והטבלאות ייווצרו).
#
# > שמות שחקנים/קבוצות מוצגים באנגלית; ההסברים בעברית.
# > כל שלב מופרד ב-Markdown משלו. הערות נוספות מופיעות גם בתוך הקוד.
#
# </div>

# %%
# --- הגדרות וסביבה ---
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# שורש הפרויקט = תיקיית האב של notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
print("project root:", ROOT)
print("processed files:", [p.name for p in PROC.glob("*.csv")])

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 1 — מבנה הנתונים ומיפוי קודים
#
# שני מקורות נתונים גולמיים. נציג את הגדלים ואת טבלת מיפוי קודי האירועים
# (שעליה מבוססות כל העמודות הבינאריות בהמשך).
#
# </div>

# %%
from data_loader import load_players_data, load_events_data

players_raw = load_players_data(RAW / "male_players.csv")   # מסונן ל-FC24 בלבד
events_raw = load_events_data(RAW / "events.csv")

print(f"FC24 players: {players_raw.shape[0]:,} שורות × {players_raw.shape[1]} עמודות")
print(f"events:       {events_raw.shape[0]:,} שורות × {events_raw.shape[1]} עמודות")
print(f"events — משחקים ייחודיים: {events_raw['id_odsp'].nunique():,} | "
      f"שחקנים: {events_raw['player'].nunique():,} | גולים: {int(events_raw['is_goal'].sum()):,}")

# %%
# טבלת מיפוי קודי event_type (מתוך docs/01_schema.md) — מוצגת כ-DataFrame
from clean_events import (EVENT_TYPE_NAMES, BODYPART_NAMES,
                          LOCATION_NAMES, IN_BOX_LOCATIONS)

map_df = pd.DataFrame(
    sorted(EVENT_TYPE_NAMES.items()), columns=["code", "event_type_name"]
)
map_df

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 2 — ניתוח חוקר (EDA)
#
# סטטיסטיקה תיאורית, אחוזי ערכים חסרים, וגרפים. נבדוק את הממצא המרכזי:
# **החוסר של 11% בסטטים הוא בגלל שוערים**, לא בעיית איכות.
#
# </div>

# %%
# סטטיסטיקה תיאורית — שחקני FC24
key_cols = ["age", "overall", "potential", "value_eur", "wage_eur",
            "pace", "shooting", "passing", "dribbling", "defending", "physic"]
desc = players_raw[key_cols].describe().T[["mean", "50%", "std", "min", "max"]]
desc = desc.rename(columns={"50%": "median"}).round(2)
desc

# %%
# אחוזי ערכים חסרים — שחקני FC24
miss = players_raw.isna().sum()
miss = (pd.DataFrame({"missing": miss, "missing_pct": (miss / len(players_raw) * 100).round(2)})
        .query("missing > 0").sort_values("missing_pct", ascending=False))
miss

# %%
# אימות: האם כל ה-NaN ב-pace שייכים לשוערים?
gk = players_raw["player_positions"].str.startswith("GK")
print(f"שוערים (GK כעמדה ראשונה): {gk.sum():,}")
print(f"מתוכם pace חסר:           {players_raw.loc[gk, 'pace'].isna().sum():,}")
print(f"שחקני שדה עם pace חסר:    {players_raw.loc[~gk, 'pace'].isna().sum():,}  ← צריך להיות 0")

# %%
# גרפים בסיסיים (4 גרפים — נטמעים במחברת)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.histplot(players_raw["age"].dropna(), bins=30, color="#2a9d8f", ax=axes[0, 0])
axes[0, 0].set_title("התפלגות גיל (Age)")

sns.histplot(players_raw["overall"].dropna(), bins=40, color="#264653", ax=axes[0, 1])
axes[0, 1].set_title("התפלגות דירוג כללי (Overall)")

val = players_raw.loc[players_raw["value_eur"] > 0, "value_eur"]
sns.histplot(val, bins=50, color="#e76f51", log_scale=(True, False), ax=axes[1, 0])
axes[1, 0].set_title("התפלגות שווי שוק (Value, סקאלת log)")

ev_counts = events_raw["event_type"].map(EVENT_TYPE_NAMES).value_counts()
sns.barplot(x=ev_counts.values, y=ev_counts.index, color="#457b9d", ax=axes[1, 1])
axes[1, 1].set_title("ספירת אירועים לפי סוג")

plt.tight_layout()
plt.show()

# %% [markdown]
# <div dir="rtl" align="right">
#
# **קריאה:** `value_eur` מוטה ימינה בקיצוניות (ממוצע ≫ חציון) — זה מצדיק גלאי
# "מציאות" בשלב 11. `overall` מתפלג כפעמון סביב 66 — מתאים לדמיון/קלאסטרינג.
#
# </div>

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 3 — ניקוי שחקנים (clean_players)
#
# טבלה נקייה ברמת שחקן + עמודות ציון מחושבות. נבדוק: אין כפילויות, אין NaN
# בעמודות מפתח, קיבוץ עמדות נכון, וניקוי שמות תקין.
#
# </div>

# %%
clean_players = pd.read_csv(PROC / "clean_players.csv")
print(f"clean_players: {clean_players.shape[0]:,} שורות × {clean_players.shape[1]} עמודות")
clean_players[["short_name", "long_name", "clean_name", "position_group",
               "age", "overall", "ability_score", "value_eur",
               "market_efficiency_score"]].head()

# %%
# בדיקות איכות — שלב 3
print("כפילויות player_id:", int(clean_players["player_id"].duplicated().sum()), " ← 0")
print("NaN ב-overall:    ", int(clean_players["overall"].isna().sum()), " ← 0")
print("NaN ב-value_eur:  ", int(clean_players["value_eur"].isna().sum()), " (סוכנים חופשיים)")
print("\nחלוקת position_group:")
print(clean_players["position_group"].value_counts(dropna=False).to_string())
print("\nממוצע ability_score לפי קבוצה:")
print(clean_players.groupby("position_group")["ability_score"].mean().round(2).to_string())

# %%
# מועמדים ל"מציאה" בקרב שחקני שדה (סיגנל גולמי; הדגל האמיתי בשלב 11)
outfield = clean_players[clean_players["position_group"] != "GK"]
cols = ["short_name", "position_group", "age", "overall",
        "ability_score", "value_eur", "market_efficiency_score"]
outfield.nlargest(8, "market_efficiency_score")[cols]

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 4 — ניקוי אירועים (clean_events)
#
# תרגום קודים לטקסט + 12 עמודות בינאריות. הקובץ גדול (~209MB) — נטען רק את
# העמודות הדרושות לבדיקה. נוודא: כל הקודים תורגמו, וסכומי הגולים מתאימים.
#
# </div>

# %%
check_cols = ["clean_name", "event_type", "event_type_name", "bodypart_name",
              "is_shot", "is_goal", "is_key_pass", "is_box_shot",
              "is_left_foot", "is_right_foot", "is_header", "is_on_target"]
clean_events = pd.read_csv(PROC / "clean_events.csv", usecols=check_cols, low_memory=False)
print(f"clean_events: {clean_events.shape[0]:,} שורות (טענו {clean_events.shape[1]} עמודות נבחרות)")

print("\nסכומי עמודות בינאריות:")
for c in ["is_shot", "is_goal", "is_key_pass", "is_box_shot", "is_on_target"]:
    print(f"  {c:<14}: {int(clean_events[c].sum()):,}")

untranslated = clean_events.loc[clean_events["event_type"].notna()
                                & clean_events["event_type_name"].isna()]
print(f"\nשורות event_type לא מתורגמות: {len(untranslated)}  ← 0")

# %%
# גולים לפי חלק גוף (בדיקת שפיות — ימין שמאל ראש)
goals_bp = clean_events.loc[clean_events["is_goal"] == 1, "bodypart_name"].value_counts(dropna=False)
print(goals_bp.to_string())

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 5 — סטטיסטיקת שחקן × משחק (player_match_stats)
#
# אגרגציה לרמת שחקן-במשחק. נוודא **התאמת גולים** מול clean_events, ונציג את
# משחקי ה-"ברייס" (2+ גולים) המובילים — אמורים להיות שחקני עילית אמיתיים.
#
# </div>

# %%
pms = pd.read_csv(PROC / "player_match_stats.csv")
print(f"player_match_stats: {pms.shape[0]:,} שורות × {pms.shape[1]} עמודות")
print(f"שחקנים ייחודיים: {pms['clean_name'].nunique():,} | משחקים: {pms['id_odsp'].nunique():,}")
pms.head()

# %%
# בדיקות איכות — שלב 5
print("סך גולים:", int(pms["goals"].sum()), " (מול 24,446 ב-clean_events; הפרש = גולים ללא שחקן)")
print("מקס' גולים לשחקן במשחק:", int(pms["goals"].max()))
print("שורות goals > shots:", int((pms["goals"] > pms["shots"]).sum()), " ← 0")
print("שורות box_shots > shots:", int((pms["box_shots"] > pms["shots"]).sum()), " ← 0")

print(f"\nמשחקי ברייס (2+ גולים): {int((pms['goals'] >= 2).sum()):,}")
pms[pms["goals"] >= 2].nlargest(5, "goals")[["clean_name", "id_odsp", "goals", "shots"]]

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 6 — אגרגציה לשחקן וציוני ביצוע (player_event_stats)
#
# שורה אחת לכל שחקן: סכומים, per_match, יחסים (0–1), וארבעה ציונים מחושבים.
# **ולידציה מרכזית:** מובילי המעורבות ההתקפית אמורים להיות חלוצי עילית אמיתיים.
#
# </div>

# %%
pes = pd.read_csv(PROC / "player_event_stats.csv")
print(f"player_event_stats: {pes.shape[0]:,} שורות × {pes.shape[1]} עמודות")
pes[["clean_name", "matches", "total_goals", "goals_per_match",
     "attacking_involvement_score", "creative_score",
     "discipline_score", "foot_balance_score"]].head()

# %%
# בדיקות איכות — שלב 6
print("התאמת goals_per_match = total_goals/matches:",
      bool(np.allclose(pes["total_goals"] / pes["matches"],
                       pes["goals_per_match"], equal_nan=True)))
for c in ["shot_accuracy", "conversion_rate", "box_shot_rate"]:
    v = pes[c].dropna()
    print(f"  {c}: min={v.min():.3f} max={v.max():.3f}  ← בטווח 0..1")
print("יחסים מעל 1:", int((pes[["shot_accuracy", "conversion_rate", "box_shot_rate"]] > 1).sum().sum()), " ← 0")
print("שחקנים עם 5+ משחקים:", int((pes["matches"] >= 5).sum()))

# %%
# ולידציה: מובילי המעורבות ההתקפית = חלוצי עילית?
cols = ["clean_name", "matches", "total_goals", "goals_per_match",
        "attacking_involvement_score", "creative_score"]
pes.nlargest(5, "attacking_involvement_score")[cols]

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 7 — הטבלה המרכזית (final_scouting_table)
#
# מיזוג פרופיל FC24 + ביצועי אירועים. **הרגע שבו מתגלה היקף ה-PoC:** כמה
# שחקנים בפועל קיימים בשני המקורות (פער השנים 2012–2017 מול 2023).
#
# </div>

# %%
final = pd.read_csv(PROC / "final_scouting_table.csv")
n = len(final); matched = int(final["has_event_data"].sum())
print(f"final_scouting_table: {n:,} שורות × {final.shape[1]} עמודות")
print(f"כפילויות player_id: {int(final['player_id'].duplicated().sum())}  ← 0")
print(f"has_event_data=True:  {matched:,}  ({matched/n*100:.1f}%)")
print(f"has_event_data=False: {n-matched:,}  ({(n-matched)/n*100:.1f}%)")

# %%
# כיסוי נתוני אירועים לפי קבוצת עמדה
cov = final.groupby("position_group")["has_event_data"].agg(["sum", "count"])
cov["pct"] = (cov["sum"] / cov["count"] * 100).round(1)
cov

# %%
# בדיקת שפיות: שחקני FC24 מובילים שהתאימו לנתוני אירועים — הסטטיסטיקה הגיונית?
cols = ["short_name", "clean_name", "overall", "has_event_data", "matches", "total_goals"]
final[final["has_event_data"]].nlargest(8, "overall")[cols]

# %% [markdown]
# <div dir="rtl" align="right">
#
# **מסקנה:** רק ~5% מהשחקנים קיימים בשני המקורות — זו המגבלה המרכזית (פער
# השנים). אבל ההתאמה **מדויקת** (Lewandowski 124 גולים, Kane 65) ו-902 שחקנים
# מספיקים בשפע לדמיון/קלאסטרינג/חריגות. החיפוש לפי פרופיל עובד על כל 18,350.
#
# </div>

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 8 — פונקציות חיפוש (search.py)
#
# 6 פונקציות חיפוש פרמטריות. כל סף הוא פרמטר. נריץ כמה דוגמאות ונוודא שהתוצאות
# הגיוניות (דמבלה כדו-רגלי, Lewandowski בברייסים).
#
# </div>

# %%
import search as S

# [1] כנפיים צעירים ומהירים (חיפוש פרופיל — על כל 18,350)
S.search_players_by_profile(final, position_group="Forward", max_age=22,
                            min_pace=85, max_value_eur=30_000_000, top_n=5)[
    ["short_name", "age", "overall", "value_eur", "preferred_foot", "has_event_data"]]

# %%
# [2] שחקנים התקפיים (ביצועי אירועים) — מובילי attacking_involvement
S.search_attacking_players(final, max_value_eur=40_000_000, top_n=5)[
    ["short_name", "age", "overall", "total_goals", "goals_per_match",
     "attacking_involvement_score"]]

# %%
# [5] דו-רגליים (20+ בעיטות) — דמבלה אמור לצוץ
S.search_two_footed_players(final, top_n=5)[
    ["short_name", "position_group", "foot_balance_score", "total_shots"]]

# %%
# [6] שחקנים עם 5+ ברייסים — Lewandowski אמור להוביל
S.find_players_with_min_braces(final, min_braces=5, top_n=5)[
    ["short_name", "total_goals", "matches_with_2_plus_goals", "goals_per_match"]]

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 9 — דמיון שחקנים ב-Cosine (similarity.py) · שימוש #1
#
# נרמול פיצ'רים (StandardScaler) + Cosine. כולל **השוואת מודלים** קריטית
# לפרק 5+6: עם קטגוריית עמדה מול בלי. דמיון בטווח 0–1.
#
# </div>

# %%
import importlib, similarity as SIM
importlib.reload(SIM)

# שחקנים הדומים ל-Kevin De Bruyne (WITH category — אותה עמדה)
sim_res, tgt = SIM.find_similar_players(final, "De Bruyne", top_n=8, same_position=True)
print(f"מטרה: {tgt['short_name']} ({tgt['position_group']}, overall {tgt['overall']})")
print("טווח דמיון:", sim_res['similarity'].min(), "..", sim_res['similarity'].max(), "← 0..1")
sim_res

# %% [markdown]
# <div dir="rtl" align="right">
#
# ### השוואת מודלים — עם קטגוריה מול בלי
# מובילי הדמיון ל-De Bruyne **בלי** הגבלת עמדה — שימו לב אם צצים שחקנים מעמדות אחרות.
#
# </div>

# %%
with_cat, without_cat, overlap = SIM.compare_with_without_category(final, "De Bruyne", top_n=8)
print(f"חפיפה בין שתי הרשימות: {overlap}/8")
without_cat[["short_name", "position_group", "overall", "similarity", "reason"]]

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 10 — קיבוץ סגנונות משחק ב-K-Means (clustering.py) · שימוש #1
#
# קיבוץ לסגנונות. בחירת **k בשיטת Elbow**. כל קלאסטר מתואר מילולית. ולידציה:
# קלאסטר העילית אמור להכיל את Messi/Mbappé/De Bruyne.
#
# </div>

# %%
import clustering as CL
importlib.reload(CL)

# שיטת Elbow — inertia + silhouette לכל k (גרף נטמע במחברת)
elbow = CL.elbow_analysis(final, save_path=None)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(elbow["k"], elbow["inertia"], "o-", color="#264653")
ax1.set_xlabel("k (מספר קלאסטרים)"); ax1.set_ylabel("Inertia", color="#264653")
ax2 = ax1.twinx()
ax2.plot(elbow["k"], elbow["silhouette"], "s--", color="#e76f51")
ax2.set_ylabel("Silhouette", color="#e76f51")
ax1.axvline(5, ls=":", color="gray"); plt.title("Elbow — נבחר k=5")
plt.tight_layout(); plt.show()

# %%
# טוענים את הטבלה עם cluster_id (נכתב ע"י clustering.py) ומתארים כל קלאסטר
clustered = pd.read_csv(PROC / "final_scouting_table.csv")
desc = pd.DataFrame([CL.describe_cluster(clustered, c) for c in range(CL.DEFAULT_K)])
desc[["cluster_id", "label", "size", "sample_players"]]

# %%
# הקלאסטר ה"טכני" — אמור להכיל את שחקני העילית
tech = desc.sort_values("size")  # just to access; show players of the elite cluster
for c in range(CL.DEFAULT_K):
    members = clustered[clustered["cluster_id"] == c]
    if {"l. messi", "k. mbappé"} & set(members["short_name"].str.lower()):
        print(f"קלאסטר העילית = {c} ({CL.describe_cluster(clustered, c)['label']})")
        break
CL.get_players_from_cluster(clustered, c, top_n=8)

# %% [markdown]
# <div dir="rtl" align="right">
#
# ### מפת הקלאסטרים — היטל PCA דו-ממדי (צבע = סגנון משחק)
# המחשה ויזואלית של הפרדת הסגנונות.
#
# </div>

# %%
fig, ax = plt.subplots(figsize=(10, 7))
CL.plot_clusters_2d(clustered, ax=ax)
plt.tight_layout(); plt.show()

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## שלב 11 — גילוי חריגות (anomaly.py) · שימוש #2
#
# (א) מציאות (יכולת גבוהה במחיר נמוך), (ב) חריגות פרופיל-מול-ביצוע. כולל
# **השוואת מודלים**: Isolation Forest מול DBSCAN מול One-Class SVM.
#
# </div>

# %%
import anomaly as AN
importlib.reload(AN)

# [2a] מציאות עילית-וזולות (min_overall=78) — בלמים אגדיים שערכם ירד
AN.detect_bargain_players(clustered, min_overall=78, top_n=8)

# %%
# [2b] חריגות פרופיל-מול-ביצוע (שחקנים עם נתוני אירועים)
AN.detect_profile_performance_anomalies(clustered, top_n=8)

# %%
# השוואת מודלים — IF מול DBSCAN מול One-Class SVM (אותו מרחב פיצ'רים)
cmp = AN.compare_if_vs_dbscan(clustered)
pd.DataFrame([cmp]).T.rename(columns={0: "count"})

# %% [markdown]
# <div dir="rtl" align="right">
#
# **מסקנה:** DBSCAN מסמן רק 14 חריגות (הנתונים 'ענן רציף' בלי פערי צפיפות) בעוד
# IF/OCSVM מסמנים ~2%. זו ולידציה שבחירת Isolation Forest נכונה לנתונים האלה.
#
# </div>

# %% [markdown]
# <div dir="rtl" align="right">
#
# ## ✅ סיכום — מצב הטבלאות
#
# טבלת סטטוס של כל קבצי העיבוד שנבנו עד כה.
#
# </div>

# %%
summary = pd.DataFrame([
    {"stage": "3", "file": "clean_players.csv", "rows": len(clean_players),
     "entity": "שחקן (FC24)"},
    {"stage": "4", "file": "clean_events.csv", "rows": len(clean_events),
     "entity": "אירוע במשחק"},
    {"stage": "5", "file": "player_match_stats.csv", "rows": len(pms),
     "entity": "שחקן × משחק"},
    {"stage": "6", "file": "player_event_stats.csv", "rows": len(pes),
     "entity": "שחקן (מצרפי)"},
    {"stage": "7", "file": "final_scouting_table.csv", "rows": len(final),
     "entity": "שחקן (טבלה מרכזית)"},
])
summary["status"] = "✅"
summary

# %% [markdown]
# <div dir="rtl" align="right">
#
# ### הצעדים הבאים (יתווספו למחברת בהמשך)
# - שלב 6 — `player_event_stats` (שורה לשחקן + ציונים מצרפיים)
# - שלב 7 — `final_scouting_table` (מיזוג פרופיל + ביצועים)
# - שלבים 8–11 — חיפוש, דמיון (Cosine), קלאסטרינג (K-Means), חריגות (Isolation Forest)
# - שלבים 12–15 — סוכן GPT, דוחות, בדיקות, ופריסה חיה
#
# </div>
