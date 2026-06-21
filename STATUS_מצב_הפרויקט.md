# 📋 Master Scout — מצב הפרויקט (מסמך מרכזי)

> מסמך חי. עודכן לאחר השלמת שלבים 12–13 (סוכן GPT + דוחות NLG). תאריך: 2026-06-17.
> שמות שחקנים/קבוצות באנגלית; ההסברים בעברית.

---

## 🎯 1. המטרה

**Master Scout** — סוכן AI לסקאוטינג (גילוי) שחקני כדורגל, פרויקט גמר לסדנת AI & ML (קורס 277302). שווה 80% מהציון.

**הרעיון המרכזי / החדשנות:** שילוב של שני מקורות שאף כלי סקאוטינג רגיל לא משלב —
1. **פרופיל סטטי** (FC24): מה השחקן "אמור" להיות (overall, pace, value...).
2. **ביצועים אמיתיים** (Football Events): מה השחקן *באמת* עשה במגרש (גולים, מסירות מפתח, כרטיסים).

המשתמש כותב בשפה חופשית (עברית/אנגלית), והסוכן: מזהה כוונה → מחלץ פרמטרים → מריץ את האלגוריתם המתאים → מחזיר דוח סקאוטינג מנומק.

**גבולות (סקופ):** שחקנים בלבד. לא קבוצות, לא חיזוי משחקים, לא הימורים. הסוכן **עוצר לפני פעולה** (ממליץ, לא "חותם" שחקן).

---

## 🗂️ 2. הדאטא שלנו

| מקור | קובץ | גודל | תוכן |
|---|---|---|---|
| EA Sports FC 24 (Kaggle) | `data/raw/male_players.csv` | 180,021 שורות, 109 עמודות | פרופיל שחקנים. 10 גרסאות FIFA — **אנחנו משתמשים ב-FC24 בלבד: 18,350 שחקנים**. |
| Football Events (Kaggle) | `data/raw/events.csv` | 941,009 שורות, 22 עמודות | אירועי משחק 2012–2017. 9,074 משחקים, 6,118 שחקנים, 24,446 גולים. |

**דרישת Big Data:** ✅ עומדים בקלות (מאות אלפי שורות, >10 עמודות, מיזוג 2 מקורות).

**פער השנים (ה-PoC):** FC24 מ-2023, האירועים מ-2012–2017. רק **902 שחקנים (4.9%)** קיימים בשני המקורות. זו המגבלה המרכזית — מתועדת בכנות (פרק 9 במחוון). חיפוש לפי פרופיל עובד על כל 18,350; ניתוחי הביצועים על 902.

---

## 🛠️ 3. הכלים והטכנולוגיות

| תחום | בחירה | תפקיד |
|---|---|---|
| שפה | Python 3.11 (conda env בשם `masterscout`) | כל הקוד |
| עיבוד נתונים | pandas, numpy | ניקוי, אגרגציה, מיזוג |
| אלגוריתמי ML | scikit-learn | Cosine, K-Means, DBSCAN, Isolation Forest, One-Class SVM |
| ויזואליזציה | matplotlib, seaborn | גרפים (EDA, Elbow, קלאסטרים) |
| **מוח הסוכן (LLM)** | OpenAI `gpt-4o-mini` | **רק** זיהוי כוונה + חילוץ פרמטרים ל-JSON + כתיבת דוח. **לא** מחשב אלגוריתמים. |
| פריסה | GitHub → Streamlit Community Cloud | קישור חי ציבורי (תנאי מעבר קריטי) |
| תיעוד | python-docx (Word), Jupyter (מחברת בדיקה) | מסמכי הגשה ובדיקה |

**"שימוש, לא אימון":** אנחנו מריצים אלגוריתמים מוכנים (`KMeans().fit()` וכו') ומסבירים את ההיגיון — בדיוק מה שהמחוון דורש.

---

## ✅ 4. מה בוצע עד כה

### תשתית (Part A)
- ✅ סביבת conda `masterscout` (Python 3.11) + `requirements.txt`
- ✅ מבנה תיקיות (`data/`, `src/`, `scripts/`, `notebooks/`, `docs/`, `reports/`)
- ✅ `.gitignore` (מפתחות וקבצים כבדים לא יעלו ל-GitHub), `.env.example`

### צינור הנתונים (שלבים 1–7) — 7 טבלאות
| שלב | מודול | פלט | תוצאה |
|---|---|---|---|
| 1 | — | `docs/01_schema.md` | מיפוי עמודות + טבלת קודי אירועים |
| 2 | `scripts/run_eda.py` | `docs/02_eda.md` + 6 גרפים | תובנה: חוסר 11% = שוערים |
| 3 | `src/clean_players.py` | `clean_players.csv` (18,350) | position_group, ability_score, market_efficiency_score |
| 4 | `src/clean_events.py` | `clean_events.csv` (941K) | מיפוי קודים + 12 עמודות בינאריות |
| 5 | `src/player_match_stats.py` | `player_match_stats.csv` (228K) | אגרגציה שחקן×משחק; גולים מתאמתים |
| 6 | `src/player_event_stats.py` | `player_event_stats.csv` (6,106) | 4 ציונים; מובילים = Ronaldo/Messi |
| 7 | `src/final_scouting_table.py` | **`final_scouting_table.csv` (18,350×78)** | **הטבלה המרכזית** + has_event_data |

### אלגוריתמי הליבה (שלבים 8–11) — **השימוש הכפול**
| שלב | מודול | מה עושה | ולידציה |
|---|---|---|---|
| 8 | `src/search.py` | 6 פונקציות חיפוש פרמטריות | Dembélé דו-רגלי, Lewandowski 35 ברייסים |
| 9 | `src/similarity.py` | **Cosine** — שחקנים דומים + השוואת מודלים (עם/בלי קטגוריה) | De Bruyne → Bruno Fernandes |
| 10 | `src/clustering.py` | **K-Means** — 5 סגנונות (Elbow) + גרף צבעוני | קלאסטר עילית = Messi/Mbappé |
| 11 | `src/anomaly.py` | **Isolation Forest + DBSCAN + One-Class SVM** — מציאות + חריגות | מציאות = Chiellini/Ramos/Thiago Silva |

> **הדרישה הכפולה של המחוון מולאה במלואה:** דמיון+קלאסטרינג → המלצה ; חריגות → דגל/מציאה.

### תיעוד ובדיקה
- ✅ `docs/ספים_והחלטות_Master_Scout.docx` — **מסמך Word** המסביר כל סף/החלטה ולמה (כולל גרפים מוטמעים). מתעדכן כל שלב.
- ✅ `notebooks/system_check.ipynb` — **מחברת בדיקת שפיות** (עברית RTL, מורצת, עם טבלאות וגרפים). מכסה שלבים 1–11.

---

## ✅ שלבים 12–13 — הושלמו (2026-06-17)
| שלב | מודול | מה עושה | ולידציה |
|---|---|---|---|
| 12 | `src/agent.py` | סוכן GPT — סיווג כוונה + חילוץ פרמטרים ל-JSON, ניתוב ל-10 הפונקציות, גבול סקופ, שאלת הבהרה | סיווג נכון he+en; סירוב לקבוצות/כרטיסים; פרשנות עקבית (מהיר→pace85, זול→bargains) |
| 13 | `src/report.py` | דוחות NLG — GPT כותב דוח קצר ומנומק בשפת השאלה, מציין מקור נתונים, שמות באנגלית | דוחות נקיים בלי המצאת מספרים; "סלאח"→Salah→Rashford/Gakpo/Mané |

**מודלים:** `gpt-4.1-nano` לסיווג (זול, הרבה איטרציות) · `gpt-4.1-mini` לדוח (איכותי יותר). שניהם משתני env. ברירת מחדל = 4 תוצאות (הערת מרצה).

| 14 | `tests/run_demo_tests.py` | 8 מקרים תקינים (PASS) + 2 **מקרי כשל מתועדים** + מקרה מחוץ-לסקופ + not_found ידידותי | מייצר `reports/14_test_results.md` לפרק 7 |

**מקרי הכשל (פרק 7):** (F1) "חלוצים מהירים שדומים למסי" — ניתוב כוונה-יחידה זורק את פילטר הפרופיל. (F2) "קבוצה של חלוצים" — סירוב-שווא בגלל המילה "קבוצה".

## ⏳ 5. מה נותר לבצע (שלב 15)

| שלב | מה | תלוי ב… |
|---|---|---|
| **15** | **פריסה חיה** (`app.py` ב-Streamlit) — צ'אט, חיבור כל השכבות, מפתח ב-Secrets, GitHub → Streamlit Cloud | 🔑 GitHub + Streamlit |

לאחר מכן: כתיבת **תיק ההגשה** (10 פרקים לפי המחוון) — אני אספק חומר ותוצאות, רון כותב/עורך.

---

## 👤 6. חלוקת תפקידים

### מה **אני (Claude Code)** אבצע
- כתיבת כל הקוד: `agent.py`, `report.py`, `app.py`, הבדיקות.
- הרצת בדיקות איכות בכל שלב + עדכון המחברת ומסמך הספים.
- אספקת חומר גולמי לתיק ההגשה (פסאודו-קוד, תוצאות, הסברים, גרפים).

### מה **אתה (רון)** צריך לספק
| # | משימה | נדרש בשלב | סטטוס |
|---|---|---|---|
| 1 | **חשבון OpenAI + מפתח API + קרדיט ~5$** | 12 | ⬜ |
| 2 | **חשבון GitHub + repo בשם `master-scout`** | 15 | ⬜ |
| 3 | **חשבון Streamlit Community Cloud** (Sign in with GitHub) | 15 | ⬜ |
| 4 | הדאטא מ-Kaggle | — | ✅ כבר יש |
| 5 | כתיבת/עריכת תיק ההגשה (Word, 10 פרקים) | בסוף | ⬜ |

> ⚠️ את מפתח ה-OpenAI **לא שולחים בצ'אט**. אני אכין קובץ `.env`, ואתה תדביק אותו שם מקומית; ב-Streamlit הוא ייכנס תחת *Settings → Secrets*.

---

## 🔍 7. איך בודקים שהכול עובד

1. **ויזואלי (מומלץ):** `conda activate masterscout` → לפתוח `notebooks/system_check.ipynb` → **Run All**. רואים טבלאות וגרפים עם הסברים בעברית.
2. **מהיר (טרמינל):** להריץ כל מודול ישירות, למשל:
   ```bash
   conda activate masterscout
   python src/search.py        # הדגמת חיפושים
   python src/similarity.py    # דמיון + השוואת מודלים
   python src/clustering.py    # K-Means + גרפים
   python src/anomaly.py       # מציאות + חריגות
   ```

---

## 📁 8. מבנה התיקייה

```
data/raw/            → male_players.csv, events.csv (לא עולים ל-GitHub)
data/processed/      → 7 הטבלאות (clean_players ... final_scouting_table)
src/                 → המודולים (data_loader, clean_*, *_stats, search, similarity, clustering, anomaly)
scripts/             → run_eda.py, build_thresholds_doc.py
notebooks/           → system_check.ipynb (+ .py מקור)
docs/                → 01_schema.md, 02_eda.md, ספים_והחלטות_Master_Scout.docx
reports/figures/     → גרפים (EDA, Elbow, קלאסטרים)
requirements.txt, .gitignore, .env.example
WORK_PLAN_MASTER_SCOUT_FOORBALL.md   → תוכנית העבודה המקורית (15 שלבים)
STATUS_מצב_הפרויקט.md                → המסמך הזה
```

---

## 🧭 הצעד הבא המיידי
**שלב 15 — פריסה חיה.** לכתוב `app.py` (Streamlit, צ'אט שמחבר agent+report), repo בשם `master-scout`, מפתח ב-Secrets, ולחבר ל-Streamlit Cloud. דורש: GitHub + Streamlit Cloud של רון.
