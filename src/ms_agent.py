"""
agent.py — Stage 12. The Agent layer (the LLM "brain").

WHAT THE LLM DOES (and ONLY this):
  1. read a free-text message (Hebrew or English),
  2. decide if it is in scope (player scouting on our data),
  3. classify it to ONE intent and extract the parameters into JSON,
  4. flag missing required params so we can ask a clarifying question.
The LLM does NOT compute similarity/clusters/anomalies — our Python modules
(search/similarity/clustering/anomaly) do that. The agent just routes.

SCOPE BOUNDARY + STOP-BEFORE-ACTION are enforced here: out-of-domain questions
are refused, and the agent only ever *recommends* (never "signs" a player).

PROVIDER ISOLATION: every call to the LLM goes through `_classify_raw()` only.
To switch from OpenAI to Gemini/anything else, change that ONE function.
"""

# מאפשר תחביר טיפוסים מודרני (dict | None) בכל גרסת פייתון
from __future__ import annotations

# ייבוא json לפענוח/בנייה של פלט ה-LLM
import json
# ייבוא os לקריאת משתני סביבה (מפתח/שמות מודלים)
import os
# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path

# ייבוא pandas לעבודה עם הטבלה המרכזית
import pandas as pd

# --- sibling modules: works both as `python src/agent.py` and as `from src.agent ...`
# מייבאים את המודולים האחים (עובד גם כחבילה וגם כסקריפט עצמאי)
try:
    from . import search, similarity, clustering, anomaly, workingset, external
except ImportError:  # run as a plain script
    # הרצה כסקריפט רגיל — ייבוא ישיר
    import search, similarity, clustering, anomaly, workingset, external

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# שורש הפרויקט (שתי רמות מעל הקובץ)
ROOT = Path(__file__).resolve().parent.parent
# נתיב הטבלה המרכזית שעליה רץ הסוכן
DATA_PATH = ROOT / "data" / "processed" / "final_scouting_table.csv"

# load .env if present (key never lives in the code)
# טוענים את קובץ .env אם קיים (המפתח לעולם לא בקוד)
try:
    # מייבאים את load_dotenv מהספרייה
    from dotenv import load_dotenv
    # טוענים את משתני הסביבה מקובץ .env
    load_dotenv(ROOT / ".env")
except ImportError:
    # אם הספרייה אינה מותקנת — ממשיכים בלעדיה
    pass

# the model name is a single config knob (env override -> easy to swap)
# שם מודל הסיווג (ניתן לעקיפה דרך משתנה סביבה)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# the conversation layer (guided questions, confirmation, tool calls) needs strong
# instruction-following, so it uses a more capable model than the legacy classifier.
# שם מודל שכבת השיחה (חזק יותר, לציות-הנחיות טוב יותר)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

# מספר התוצאות בברירת מחדל (סוכן מחזיר 3–4, לא ערך ויקיפדיה)
DEFAULT_TOP_N = 4  # lecturer: an agent returns 3-4 results, not a wikipedia dump


class GoalkeeperAnalysisUnavailable(Exception):
    """Raised when the user enters a goalkeeper's name. Goalkeepers have no outfield
    attributes in our data, so play-style analysis (similarity/clustering) can't run
    yet. We answer with the goalkeeper's BASIC details only (name, age, overall,
    potential) plus a notice that fuller analysis is coming in a future update —
    instead of a broken/empty statistical result. The basic facts are carried in
    `.facts` so the notice can show them."""

    # אתחול החריגה עם מילון הפרטים הבסיסיים של השוער
    def __init__(self, facts: dict | None = None):
        # קוראים לאתחול האב עם הודעה פנימית
        super().__init__("goalkeeper analysis unavailable")
        # שומרים את הפרטים הבסיסיים (או מילון ריק)
        self.facts = facts or {}


# פונקציית עזר: שולפת רק את הפרטים הבסיסיים (שם/גיל/דירוג/פוטנציאל) משורת שחקן
def _basic_facts(row) -> dict:
    """Pull only the BASIC profile fields (name/age/overall/potential) from a player
    row — works for a pandas Series (our data) or a dict (external lookup)."""
    # פונקציה פנימית: מחזירה ערך עמודה אם קיים ולא ריק
    def g(k):
        # תומך גם ב-Series וגם ב-dict (שניהם בעלי .get)
        v = row.get(k) if hasattr(row, "get") else None
        # מחזיר את הערך רק אם הוא קיים ואינו NaN
        return v if (v is not None and pd.notna(v)) else None

    # פונקציה פנימית: ממירה ערך למספר שלם אם קיים
    def as_int(k):
        # מקבלים את הערך
        v = g(k)
        # ממירים ל-int או מחזירים None
        return int(v) if v is not None else None

    # שם השחקן: קצר אם יש, אחרת מלא
    sn = g("short_name") or g("long_name")
    # מחזירים מילון עם 4 הפרטים הבסיסיים בלבד
    return {"short_name": str(sn) if sn is not None else None,
            "age": as_int("age"), "overall": as_int("overall"),
            "potential": as_int("potential")}


# פונקציה: בונה את הודעת השוער — פרטים בסיסיים + שורת "יתווסף בעתיד", בשפת המשתמש
def _gk_notice(language: str = "he", facts: dict | None = None) -> str:
    """The fixed, honest goalkeeper notice: BASIC details (name/age/overall/potential)
    — and nothing more — followed by the 'analysis coming in a future update' line."""
    # ברירת מחדל למילון פרטים ריק
    facts = facts or {}
    # ענף עברית
    if language == "he":
        # שורת הכותרת עם הפרטים הבסיסיים (תיבנה אם יש שם)
        head = ""
        # אם יש שם שחקן — בונים את שורת הפרטים
        if facts.get("short_name"):
            # רשימת פרטים (גיל/דירוג/פוטנציאל)
            det = []
            # מוסיפים גיל אם קיים
            if facts.get("age") is not None:
                det.append(f"גיל {facts['age']}")
            # מוסיפים דירוג כללי אם קיים
            if facts.get("overall") is not None:
                det.append(f"דירוג כללי {facts['overall']}")
            # מוסיפים פוטנציאל אם קיים
            if facts.get("potential") is not None:
                det.append(f"פוטנציאל {facts['potential']}")
            # בונים את שורת הכותרת עם שם השוער
            line = f"🧤 **{facts['short_name']}** — שוער"
            # מצרפים את הפרטים אם קיימים
            if det:
                line += " · " + " · ".join(det)
            # מסיימים את הכותרת בשתי שורות חדשות
            head = line + "\n\n"
        # מחזירים את הכותרת + הודעת המגבלה
        return head + (
            "כרגע אני יכול להציג עבור שוערים את הפרטים הבסיסיים בלבד (שם, גיל, דירוג, "
            "פוטנציאל). הניתוחים הסטטיסטיים (דמיון סגנון משחק וקיבוץ) מתבצעים על "
            "**שחקני שדה בלבד**, כי לשוערים אין במאגר את 6 תכונות הליבה — זו מגבלת "
            "דאטא ידועה. **בגרסת עדכון עתידית של המערכת יתווסף ניתוח סטטיסטי מלא גם "
            "לשוערים.**"
        )
    # ענף אנגלית — אותו מבנה כמו ענף העברית
    head = ""
    # אם יש שם שחקן — בונים שורת פרטים
    if facts.get("short_name"):
        # רשימת פרטים
        det = []
        # גיל
        if facts.get("age") is not None:
            det.append(f"age {facts['age']}")
        # דירוג כללי
        if facts.get("overall") is not None:
            det.append(f"overall {facts['overall']}")
        # פוטנציאל
        if facts.get("potential") is not None:
            det.append(f"potential {facts['potential']}")
        # שורת כותרת עם שם השוער
        line = f"🧤 **{facts['short_name']}** — goalkeeper"
        # מצרפים פרטים אם קיימים
        if det:
            line += " · " + " · ".join(det)
        # מסיימים בשתי שורות חדשות
        head = line + "\n\n"
    # מחזירים כותרת + הודעת המגבלה באנגלית
    return head + (
        "For goalkeepers I can currently show only the basic details (name, age, "
        "overall, potential). Statistical analyses (play-style similarity and "
        "clustering) run on **outfield players only**, because goalkeepers don't "
        "have the 6 core attributes in our data — a known data limitation. **A future "
        "system update will add full statistical analysis for goalkeepers too.**"
    )

# intents the agent can route to (name -> short human description, for docs/tests)
# מילון הכוונות שהסוכן יכול לנתב אליהן (שם → תיאור קצר, לתיעוד/בדיקות)
INTENTS = {
    # חיפוש לפי פרופיל FC24
    "profile_search": "filter by FC24 profile (position, age, overall, pace, value, foot, league, nationality)",
    # חלוצים מדורגים לפי תפוקה התקפית אמיתית
    "attacking_players": "attackers ranked by real attacking output (event data)",
    # קשרים מדורגים לפי יצירת הזדמנויות
    "creative_midfielders": "midfielders ranked by chance creation (event data)",
    # מגנים מדורגים לפי משמעת (מעט כרטיסים)
    "disciplined_defenders": "defenders ranked by low-card discipline (event data)",
    # שחקנים דו-רגליים
    "two_footed": "players who shoot well with both feet (event data)",
    # שחקנים עם לפחות N משחקי דאבל
    "braces": "players with at least N two-goal games (event data)",
    # שחקנים דומים בסגנון לשחקן נתון (Cosine)
    "similar_players": "players similar in play-style to a NAMED player (Cosine)",
    # קיבוץ לסגנונות משחק (K-Means), אופציונלית בתוך עמדה
    "clustering": "group players into play-styles / describe the styles (K-Means), optionally within one position",
    # מציאות: יכולת גבוהה מול שווי נמוך (חריגות)
    "bargains": "underpriced players: high ability vs low value (anomaly)",
    # אי-התאמה בין דירוג FC24 לתפוקה אמיתית (חריגות)
    "profile_performance_anomaly": "players whose FC24 rating disagrees with real output (anomaly)",
    # ויזואליזציה של התוצאות האחרונות (פעולת המשך)
    "visualize": "chart the MOST RECENT results (a follow-up, not a new search)",
    # ברכה / שיחת חולין
    "greeting": "the user is only greeting / making small talk (hi, hello, מה קורה)",
    # איפוס קבוצת העבודה והתחלה מחדש
    "reset": "clear the current working set and start a fresh search from the full table",
}

# ---------------------------------------------------------------------------
# The system prompt — the heart of the agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are "Master Scout", an AI football (soccer) player-scouting agent.

You do NOT compute anything yourself. Your only job is to read the user's free-text
message (Hebrew or English) and translate it into a SINGLE structured JSON command
that a Python backend will execute.

SCOPE — you handle ONLY individual PLAYER scouting questions answerable from our two
datasets (EA Sports FC24 player profiles + real match-event performance). You MUST
refuse (set in_scope=false) anything else, including: team or tactical analysis,
match/score prediction, betting or odds, transfer paperwork, tickets, news, general
knowledge, or any non-football topic. You never take real-world action — you only
recommend; if asked to actually sign/buy/transfer a player, refuse.
When refusing, write a short polite refusal in `refusal`, IN THE USER'S LANGUAGE.

INTENTS — choose EXACTLY ONE for `intent`:
- profile_search: filter players by FC24 profile attributes.
- attacking_players: attackers ranked by real attacking output.
- creative_midfielders: midfielders ranked by chance creation.
- disciplined_defenders: defenders ranked by discipline (few cards).
- two_footed: players who shoot well with both feet.
- braces: players with at least N two-goal games. Triggers: "braces" /
  "צמדי גולים" / "משחקי דאבל" / "דאבלים" / "two-goal games". This is a per-PLAYER
  stat and is ALWAYS in scope — never refuse it as a "team statistic".
- similar_players: players most similar in play-style to a NAMED player.
- clustering: group players into play-styles. Can be ALL players, or within ONE
  position if the user names one ("חלק את החלוצים ל-4 סגנונות" -> position_group=Forward, n_clusters=4).
- bargains: underpriced players (high ability vs low market value).
- profile_performance_anomaly: players whose FC24 rating disagrees with real output.
- visualize: the user wants a CHART of the results from the PREVIOUS turn
  ("ויזואליזציה", "תעשה גרף", "visualize that", "chart this", "תרשים"). This is a
  follow-up; keep filters empty.
- greeting: the user is ONLY greeting or making small talk and has not asked for
  anything yet ("היי", "מה קורה", "מה נשמע", "hi", "hello", "תודה"). Set
  in_scope=true, intent="greeting", filters empty. Do NOT refuse a greeting.
- reset: the user wants to clear the current narrowed list and start fresh
  ("התחל מחדש", "מהתחלה", "אפס", "נקה", "start over", "reset", "new search").
  Set in_scope=true, intent="reset", filters empty.

FILTERS — include ONLY the ones the user actually specified; OMIT all others:
  position_group : one of "Forward","Midfielder","Defender","GK"
  max_age, min_age : integer
  min_overall, min_potential, min_pace : integer 0-99
  max_value_eur, min_value_eur : integer euros (e.g. "30 million" -> 30000000)
  preferred_foot : "Left" or "Right"
  league_name, nationality : string in English
  top_n : integer (how many results; default 10)
  player_name : string in English  -- REQUIRED for similar_players
  min_braces : integer  -- two-goal games
  min_total_goals : integer  -- total goals scored
  min_key_passes : integer  -- key passes (≈ assists / chance creation)
  min_shots : integer  -- total shots
  max_yellow_cards : integer  -- few cards (disciplined)
  min_foot_balance : integer  -- for two_footed
  n_clusters : integer  -- for clustering
  contamination : float 0-1  -- for the two anomaly intents

FOOTBALL VOCABULARY (map Hebrew/English words to the metric above):
  "גול"/"גולים"/"שער"/"שערים"/"הבקעה"/"הבקעות"/"goals" -> min_total_goals
  "בישול"/"בישולים"/"אסיסט"/"אסיסטים"/"מסירת מפתח"/"assist" -> min_key_passes
  "בעיטה"/"בעיטות"/"shots" -> min_shots
  "צמד"/"צמדי גולים"/"דאבל"/"דאבלים"/"brace" -> min_braces
  "כרטיס"/"כרטיסים"/"כרטיס צהוב" with "מעט"/"few" -> intent disciplined_defenders
Comparators: "יותר מ-"/"מעל"/"X+"/"לפחות"/"more than"/"over" before a number ->
  a MIN threshold (e.g. "יותר מ-5 גולים" -> min_total_goals: 5; "מעל 10 בישולים"
  -> min_key_passes: 10). These are all in scope (per-player stats). For a pure
  goals/assists/shots filter with no other intent, use intent "profile_search".

INTERPRETATION RULES — apply CONSISTENTLY, every time. Translate every adjective
into its numeric filter; never leave an adjective unmapped:
  "fast" / "מהיר" / "מהירים"                 -> min_pace: 85
  "very fast" / "מהיר מאוד"                  -> min_pace: 90
  "young" / "צעיר" / "צעירים"                -> max_age: 23
  "veteran" / "experienced" / "ותיק" / "מנוסה" -> min_age: 30
  "cheap" / "bargain" / "זול" / "מציאה"      -> intent "bargains" (or a low max_value_eur if a number is named)
  "expensive" / "יקר"                        -> min_value_eur: 50000000
  "elite" / "world class" / "עילית" / "ברמה עולמית" -> min_overall: 85
  "talent" / "wonderkid" / "כשרון" / "עתיד"  -> min_potential: 85 AND max_age: 21
A named result count ("top 5", "5 שחקנים", "תן 3", "give me 3") -> top_n; otherwise OMIT top_n.
Money: "30 million" / "30 מיליון" / "30M" / "€30m" -> 30000000.
These mappings are mandatory: if the user says "fast", min_pace MUST appear.

FOLLOW-UPS — a CONVERSATION_CONTEXT block (with PREVIOUS_COMMAND) may appear before
the user's message. Phrases like "מתוך אלה" / "מתוך הרשימה" / "מי מהם" / "of those"
/ "from this list" mean the user is DRILLING DOWN into the current list — treat the
message as a normal in-scope query (classify its intent + filters as usual); do NOT
refuse it. If the message just tweaks the previous results — "expand to 10" /
"הרחב ל-10" / "show more" / "עוד" / "and only under 25M" / "ורק מתחת ל-25 מיליון" /
"make them younger" / "גם צעירים יותר" — START from PREVIOUS_COMMAND and apply the
change (same intent, keep existing filters, add/replace what changed). For a pure
chart request use intent "visualize".

MISSING — the ONLY ever-required parameter is player_name for the similar_players
intent. Every other parameter has a sensible default, so `missing` MUST be []
unless the intent is similar_players and no player name was given (then set
missing=["player_name"]). Never put n_clusters, top_n, position_group, contamination
or any other field in `missing`.

LANGUAGE — set `language` to "he" if the user wrote mainly Hebrew, else "en".

NAMES — return all player/team/league/nationality names in ENGLISH, transliterating
from Hebrew if needed (e.g. "מסי"->"Messi", "ריאל מדריד"->"Real Madrid",
"ספרד"->"Spain").

Return ONLY a JSON object with EXACTLY these keys and nothing else:
{
  "in_scope": true/false,
  "intent": "<one intent name>" or null,
  "filters": { ... only specified filters ... },
  "missing": [ "<field>", ... ] or [],
  "language": "he" or "en",
  "refusal": "<refusal text if in_scope is false, else empty string>"
}

EXAMPLES:
User: "מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו, תן 5"
-> {"in_scope":true,"intent":"profile_search","filters":{"position_group":"Forward","max_age":22,"min_pace":85,"max_value_eur":30000000,"top_n":5},"missing":[],"language":"he","refusal":""}

User: "who plays like De Bruyne?"
-> {"in_scope":true,"intent":"similar_players","filters":{"player_name":"De Bruyne"},"missing":[],"language":"en","refusal":""}

User: "תמצא לי שחקנים דומים"
-> {"in_scope":true,"intent":"similar_players","filters":{},"missing":["player_name"],"language":"he","refusal":""}

User: "find me undervalued defenders over 80 rated"
-> {"in_scope":true,"intent":"bargains","filters":{"position_group":"Defender","min_overall":80},"missing":[],"language":"en","refusal":""}

User: "כמה כרטיסים צהובים קיבלה ריאל מדריד העונה?"
-> {"in_scope":false,"intent":null,"filters":{},"missing":[],"language":"he","refusal":"אני מתמקד בניתוח שחקנים בודדים בלבד ולא בנתוני קבוצות. אשמח לעזור למצוא שחקנים לפי פרופיל, דמיון, סגנון משחק או מציאות."}

User: "show me 3 elite fast forwards"
-> {"in_scope":true,"intent":"profile_search","filters":{"position_group":"Forward","min_overall":85,"min_pace":85,"top_n":3},"missing":[],"language":"en","refusal":""}

User: "מצא לי כשרונות צעירים בהגנה"
-> {"in_scope":true,"intent":"profile_search","filters":{"position_group":"Defender","min_potential":85,"max_age":21},"missing":[],"language":"he","refusal":""}

User: "cheap forwards"
-> {"in_scope":true,"intent":"bargains","filters":{"position_group":"Forward"},"missing":[],"language":"en","refusal":""}

User: "חלק את החלוצים ל-4 סגנונות"
-> {"in_scope":true,"intent":"clustering","filters":{"position_group":"Forward","n_clusters":4},"missing":[],"language":"he","refusal":""}

(CONVERSATION_CONTEXT present) PREVIOUS_COMMAND: intent=similar_players filters={"player_name":"Messi","top_n":4}
User: "הרחב ל-10"
-> {"in_scope":true,"intent":"similar_players","filters":{"player_name":"Messi","top_n":10},"missing":[],"language":"he","refusal":""}

(CONVERSATION_CONTEXT present) PREVIOUS_COMMAND: intent=profile_search filters={"position_group":"Forward","min_age":25,"max_age":30}
User: "מתוך אלה מי עם יותר מ-15 צמדי גולים"
-> {"in_scope":true,"intent":"braces","filters":{"min_braces":15},"missing":[],"language":"he","refusal":""}
"""

# ---------------------------------------------------------------------------
# LLM call — the ONLY place that talks to the provider (swap here to change LLM)
# ---------------------------------------------------------------------------
# משתנה גלובלי המאחסן את לקוח ה-LLM (נבנה פעם אחת, בעצלתיים)
_client = None


# פונקציה: בונה (פעם אחת) את לקוח OpenAI — המפתח מ-.env או מ-Streamlit secrets
def _get_client():
    """Lazily build the OpenAI client. Key from env (.env) or Streamlit secrets."""
    # מצהירים על שימוש במשתנה הגלובלי
    global _client
    # אם הלקוח כבר נבנה — מחזירים אותו
    if _client is not None:
        return _client

    # קוראים את המפתח ממשתני הסביבה
    key = os.getenv("OPENAI_API_KEY")
    # אם אין מפתח (או ערך placeholder) — מנסים את secrets של Streamlit
    if not key or key == "PASTE_YOUR_KEY_HERE":
        try:  # on Streamlit Cloud the key lives in st.secrets
            # מייבאים streamlit רק כאן (לא תמיד זמין)
            import streamlit as st
            # שולפים את המפתח מ-secrets
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            # אם אין streamlit/secrets — אין מפתח
            key = None
    # אם עדיין אין מפתח — שגיאה ברורה למשתמש
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Paste your key into the .env file "
            "(OPENAI_API_KEY=sk-...), or add it to Streamlit secrets."
        )

    # מייבאים את מחלקת הלקוח של OpenAI
    from openai import OpenAI
    # בונים את הלקוח עם המפתח
    _client = OpenAI(api_key=key)
    # מחזירים אותו
    return _client


# פונקציה: מרכיבה בלוק הקשר עם הפקודה הקודמת כדי שה-LLM יפתור פעולות-המשך
def _context_block(context: dict | None) -> str:
    """Render the previous command so the LLM can resolve follow-ups."""
    # אין הקשר — מחרוזת ריקה
    if not context:
        return ""
    # הכוונה הקודמת
    prev_intent = context.get("prev_intent")
    # אם אין כוונה קודמת — מחרוזת ריקה
    if not prev_intent:
        return ""
    # המסננים הקודמים כ-JSON (תומך עברית)
    prev_filters = json.dumps(context.get("prev_filters") or {}, ensure_ascii=False)
    # מחזירים את בלוק ההקשר לפני הודעת המשתמש
    return (f"CONVERSATION_CONTEXT (for follow-ups only):\n"
            f"PREVIOUS_COMMAND: intent={prev_intent} filters={prev_filters}\n\n"
            f"USER MESSAGE:\n")


# פונקציה: קריאת LLM יחידה — טקסט חופשי → JSON של כוונה/מסננים (מילון גולמי)
def _classify_raw(query: str, context: dict | None = None) -> dict:
    """Single LLM call: free text -> intent/filters JSON. Returns a raw dict.

    `context` (optional) carries the previous command so the model can resolve
    follow-ups like "expand to 10" or "visualize that".
    """
    # מקבלים את לקוח ה-LLM
    client = _get_client()
    # קוראים ל-LLM עם הנחיית המערכת והודעת המשתמש (עם בלוק ההקשר), בפורמט JSON
    resp = client.chat.completions.create(
        model=MODEL,
        # טמפרטורה 0 לעקביות מלאה
        temperature=0,
        # מאלצים פלט JSON תקין
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _context_block(context) + query},
        ],
    )
    # מפענחים את ה-JSON שהוחזר ומחזירים מילון
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# Public agent API
# ---------------------------------------------------------------------------
# פונקציה ציבורית: קוראת ל-LLM ומנרמלת את התוצאה לסכמה מובטחת
def classify_and_extract(query: str, context: dict | None = None) -> dict:
    """Call the LLM and normalize the result to a guaranteed schema."""
    # מקבלים את המילון הגולמי מה-LLM
    raw = _classify_raw(query, context)
    # מחזירים מילון מנורמל עם כל המפתחות הצפויים
    return {
        # האם בתחום
        "in_scope": bool(raw.get("in_scope", False)),
        # הכוונה שזוהתה
        "intent": raw.get("intent"),
        # המסננים שחולצו
        "filters": dict(raw.get("filters") or {}),
        # שדות חסרים (אם יש)
        "missing": list(raw.get("missing") or []),
        # שפת המשתמש
        "language": raw.get("language", "he"),
        # טקסט סירוב (אם מחוץ לתחום)
        "refusal": raw.get("refusal", ""),
    }


# מילון תרגום שדות חסרים להסבר בעברית (לשאלת הבהרה)
_CLARIFY_FIELDS_HE = {
    "player_name": "שם השחקן להשוואה",
    "min_braces": "כמה משחקי דאבל (2+ גולים) לפחות",
    "position_group": "עמדה (חלוץ/קשר/הגנה)",
}
# מילון תרגום שדות חסרים להסבר באנגלית
_CLARIFY_FIELDS_EN = {
    "player_name": "the player's name to compare against",
    "min_braces": "the minimum number of two-goal games",
    "position_group": "a position (Forward/Midfielder/Defender)",
}


# פונקציה: הופכת רשימת שדות חסרים לשאלת הבהרה קצרה אחת
def ask_clarifying_question(missing: list, language: str = "he") -> str:
    """Turn a list of missing fields into one short clarifying question."""
    # ענף עברית
    if language == "he":
        # מתרגמים כל שדה חסר להסבר בעברית
        parts = [_CLARIFY_FIELDS_HE.get(m, m) for m in missing]
        # בונים את שאלת ההבהרה
        return "כדי להמשיך אני צריך עוד פרט: " + ", ".join(parts) + ". תוכל להוסיף?"
    # ענף אנגלית: מתרגמים כל שדה חסר
    parts = [_CLARIFY_FIELDS_EN.get(m, m) for m in missing]
    # בונים את שאלת ההבהרה באנגלית
    return "To continue I need a bit more: " + ", ".join(parts) + ". Could you add that?"


# פונקציה: הודעת הסירוב הכללית (כשהבקשה מחוץ לתחום) בשפת המשתמש
def _default_refusal(language: str = "he") -> str:
    # ענף עברית
    if language == "he":
        return ("אני סוכן סקאוטינג לשחקנים בלבד — אני לא עוסק בקבוצות, חיזוי משחקים, "
                "הימורים או נושאים מחוץ לכדורגל. אשמח לעזור למצוא שחקנים לפי פרופיל, "
                "דמיון, סגנון משחק או מציאות.")
    return ("I'm a player-scouting agent only — I don't handle teams, match "
            "prediction, betting or non-football topics. I can help find players "
            "by profile, similarity, play-style or bargains.")


def greeting_message(language: str = "he") -> str:
    """Warm intro + capability list + example queries. Used for the chat welcome
    and as the reply when the user just says hi."""
    # ענף עברית — מחזיר ברכת פתיחה + רשימת יכולות + דוגמאות
    if language == "he":
        # מחרוזת הברכה בעברית
        return (
            "היי! אני **Master Scout** ⚽ — סוכן הסקאוטינג החכם שלך. אני משלב את "
            "הפרופיל הרשמי (EA FC24) עם ביצועים אמיתיים מהמגרש, ויכול לעזור לך ב:\n\n"
            "1. 🔎 **חיפוש לפי פרופיל** — עמדה, גיל, מהירות, שווי\n"
            "2. 👥 **שחקנים דומים** לשחקן שתבחר (Cosine)\n"
            "3. 🧩 **קיבוץ לפי סגנון משחק** — גם בתוך עמדה (K-Means)\n"
            "4. 💎 **מציאות** — יכולת גבוהה מול שווי נמוך\n"
            "5. 📈 **ביצועים אמיתיים** — גולים, דאבלים, יצירתיות, משמעת\n\n"
            "נסה משהו כמו:\n"
            "- *\"מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו\"*\n"
            "- *\"מי דומה ל-Lionel Messi?\"*\n"
            "- *\"חלק את הקשרים ל-3 סגנונות משחק\"*\n"
            "- *\"מצא מציאות בהגנה מעל דירוג 80\"*\n\n"
            "אז על מה נצא לדרך?"
        )
    # ענף אנגלית — אותה ברכה באנגלית
    return (
        "Hi! I'm **Master Scout** ⚽ — your AI scouting agent. I blend the official "
        "FC24 profile with real on-pitch performance, and can help you with:\n\n"
        "1. 🔎 **Profile search** — position, age, pace, value\n"
        "2. 👥 **Similar players** to one you name (Cosine)\n"
        "3. 🧩 **Play-style clustering** — even within a position (K-Means)\n"
        "4. 💎 **Bargains** — high ability vs low value\n"
        "5. 📈 **Real performance** — goals, braces, creativity, discipline\n\n"
        "Try something like:\n"
        "- *\"find fast forwards under 23, up to €30M\"*\n"
        "- *\"who plays like Lionel Messi?\"*\n"
        "- *\"split the midfielders into 3 styles\"*\n"
        "- *\"find undervalued defenders rated over 80\"*\n\n"
        "So, where shall we start?"
    )


# --- routing helpers --------------------------------------------------------
# פונקציית עזר: שומרת רק את המפתחות המותרים עבור פונקציית backend מסוימת
def _pick(filters: dict, keys) -> dict:
    """Keep only the allowed keys for a given backend function."""
    # בונה מילון רק מהמפתחות שקיימים ואינם None
    return {k: filters[k] for k in keys if k in filters and filters[k] is not None}


# קבוצת מפתחות מסנני הפרופיל המותרים
_PROFILE_KEYS = ("position_group", "max_age", "min_age", "min_overall",
                 "min_potential", "min_pace", "max_value_eur", "min_value_eur",
                 "preferred_foot", "league_name", "nationality")


def route_query(intent: str, filters: dict, df: pd.DataFrame):
    """Dispatch an intent+filters to the matching backend function.

    Returns (result_df, extra) where extra is a dict with anything beyond the
    main table (e.g. cluster descriptions, the similarity target name).
    Raises ValueError on an unknown intent or a bad parameter (e.g. player not found).
    """
    # עותק של המסננים (בטוח לשינוי)
    f = dict(filters or {})
    # מספר התוצאות המבוקש (ברירת מחדל DEFAULT_TOP_N)
    top_n = int(f.get("top_n", DEFAULT_TOP_N))
    # מילון מידע נוסף שיוחזר לצד הטבלה
    extra = {}

    # all the filter/search intents go through ONE generic engine so that ANY
    # recognized filter works (free-text robustness) and display == working set.
    # כל כוונות הסינון/חיפוש עוברות דרך מנוע גנרי אחד (חוסן מול טקסט חופשי)
    if intent in ("profile_search", "attacking_players", "creative_midfielders",
                  "disciplined_defenders", "two_footed", "braces"):
        # מריצים את מנוע החיפוש הגנרי
        res, _full = workingset.search(df, intent, f, top_n=top_n)

    # כוונת דמיון — דורשת שם שחקן
    elif intent == "similar_players":
        # שם השחקן להשוואה
        name = f.get("player_name")
        # אם אין שם — שגיאה
        if not name:
            raise ValueError("similar_players requires player_name")
        # מריצים דמיון Cosine ומקבלים תוצאות + שורת היעד
        res, target = similarity.find_similar_players(
            df, name, top_n=top_n,
            **_pick(f, ("same_position", "max_age", "max_value_eur")))
        # שומרים את שם היעד למידע הנוסף
        extra["target"] = str(target.get("short_name", name))

    # כוונת קיבוץ (K-Means)
    elif intent == "clustering":
        # per-position clustering: if a position is named, cluster only within it
        # אם צוינה עמדה — מקבצים רק בתוכה
        pos = f.get("position_group")
        # מצמצמים את הטבלה לעמדה אם צוינה
        df_c = df[df["position_group"] == pos].copy() if pos else df
        # let the algorithm pick the optimal K (3-6) unless the user named one
        # אם המשתמש ציין K — משתמשים בו; אחרת האלגוריתם בוחר אופטימלי
        if f.get("n_clusters"):
            n_clusters, sil = int(f["n_clusters"]), None
        else:
            # האלגוריתם בוחר K אופטימלי לפי סילואט
            n_clusters, sil = clustering.best_k(df_c)
        # מריצים את הקיבוץ ומקבלים טבלה עם cluster_id
        labeled, _model, _scaler = clustering.run_player_kmeans(df_c, n_clusters=n_clusters)
        # מזהי הקלאסטרים (מלבד -1)
        ids = sorted(c for c in labeled["cluster_id"].unique() if c != -1)
        # מתארים כל קלאסטר עם תווית ייחודית
        descriptions = clustering.describe_clusters(labeled, ids)  # distinct labels
        # one value per cell: split the dominant traits and sample players into
        # their own columns (trait_1..5, player_1..5)
        # בונים שורת תצוגה לכל קלאסטר — ערך אחד בכל תא
        rows = []
        # עוברים על תיאורי הקלאסטרים
        for d in descriptions:
            # שורת בסיס: מזהה, תווית וגודל
            row = {"cluster_id": d["cluster_id"], "label": d["label"], "size": d["size"]}
            # מפצלים את 5 התכונות הדומיננטיות לעמודות נפרדות
            for i, t in enumerate(d["dominant_traits"][:5], 1):
                row[f"trait_{i}"] = f"{t['trait']} {t['mean']:.0f}"
            # מפצלים את 5 השחקנים לדוגמה לעמודות נפרדות
            for i, name in enumerate(d["sample_players"][:5], 1):
                row[f"player_{i}"] = name
            # מוסיפים את השורה
            rows.append(row)
        # בונים טבלת תצוגה מהשורות
        res = pd.DataFrame(rows)
        # שומרים את התיאורים המלאים
        extra["descriptions"] = descriptions
        # שומרים את הטבלה המתויגת המלאה (לצמצום עתידי)
        extra["labeled"] = labeled            # full rows + cluster_id, for drilling
        # מספר הקלאסטרים בפועל
        extra["k"] = n_clusters
        # האם האלגוריתם בחר את K (True אם sil קיים)
        extra["auto_k"] = sil is not None     # True if the algorithm chose K
        # ציון הסילואט (אם נבחר אוטומטית)
        extra["silhouette"] = sil
        # אם צוינה עמדה — שומרים אותה
        if pos:
            extra["position"] = pos

    # כוונת מציאות (Isolation Forest)
    elif intent == "bargains":
        # מזהים שחקני מציאות
        res = anomaly.detect_bargain_players(
            df, top_n=top_n, **_pick(f, ("contamination", "min_overall",
                                         "max_age", "max_value_eur")))

    # כוונת אי-התאמה פרופיל-ביצועים
    elif intent == "profile_performance_anomaly":
        # מזהים שחקנים עם אי-התאמה בין דירוג לתפוקה
        res = anomaly.detect_profile_performance_anomalies(
            df, top_n=top_n, **_pick(f, ("contamination",)))

    # כוונה לא מוכרת — שגיאה
    else:
        raise ValueError(f"unknown intent: {intent!r}")

    # מחזירים את הטבלה ואת המידע הנוסף
    return res, extra


def run_agent(query: str, df: pd.DataFrame | None = None,
              context: dict | None = None) -> dict:
    """End-to-end: free text -> classify -> (refuse | clarify | route) -> result.

    `context` carries the previous command (intent + filters) so the agent can
    resolve follow-ups ("expand to 10", "visualize that").

    Returns a dict the report layer (Stage 13) / the app (Stage 15) can render:
      status: "ok" | "out_of_scope" | "clarify" | "visualize" | "not_found" | "error"
      message: refusal / clarifying question / error text (for non-"ok" statuses)
      intent, filters, language
      result: a DataFrame (for status "ok"), else None
      extra: dict of supporting info (cluster descriptions, similarity target...)
    """
    # אם לא הועברה טבלה — טוענים את הטבלה המרכזית
    if df is None:
        df = load_table()
    # מנסים לסווג ולחלץ פרמטרים מהשאילתה
    try:
        parsed = classify_and_extract(query, context)
    except RuntimeError as e:  # no API key
        # אין מפתח API — מחזירים שגיאה ידידותית
        return {"query": query, "intent": None, "filters": {}, "language": "he",
                "result": None, "extra": {}, "status": "error", "message": str(e)}
    except Exception as e:  # network / parsing
        # שגיאת רשת/פענוח — מחזירים שגיאה
        return {"query": query, "intent": None, "filters": {}, "language": "he",
                "result": None, "extra": {}, "status": "error",
                "message": f"agent error: {e}"}
    # מריצים את הפקודה שסווגה
    return execute_parsed(parsed, df, query)


# פונקציה: מריצה פקודה שכבר סווגה (מאפשרת לבדוק את parsed לפני, בלי קריאה חוזרת ל-LLM)
def execute_parsed(parsed: dict, df: pd.DataFrame, query: str = "") -> dict:
    """Run an already-classified command (lets the caller inspect `parsed` first,
    e.g. to ask a guided question, without re-calling the LLM)."""
    # מילון בסיס לתוצאה
    base = {"query": query, "intent": None, "filters": {}, "language": "he",
            "result": None, "extra": {}}
    # מעדכנים את הבסיס בכוונה, המסננים והשפה שסווגו
    base.update(intent=parsed["intent"], filters=parsed["filters"],
                language=parsed["language"])

    # "visualize" is a follow-up handled by the caller (it charts the last result)
    # ויזואליזציה — פעולת המשך שמטופלת בצד הקורא
    if parsed["intent"] == "visualize":
        return {**base, "status": "visualize", "message": ""}

    # a plain greeting -> warm intro (never refuse small talk)
    # ברכה — מחזירים פתיחה חמה (לעולם לא מסרבים לשיחת חולין)
    if parsed["intent"] == "greeting":
        return {**base, "status": "greeting",
                "message": greeting_message(parsed["language"])}

    # reset -> the caller clears the working set
    # איפוס — הצד הקורא מנקה את קבוצת העבודה
    if parsed["intent"] == "reset":
        return {**base, "status": "reset", "message": ""}

    # מחוץ לתחום — מחזירים סירוב מנומס
    if not parsed["in_scope"]:
        return {**base, "status": "out_of_scope",
                "message": parsed["refusal"] or _default_refusal(parsed["language"])}

    # only similar_players has a truly required field (player_name); never block
    # other intents on "missing" — they all have sensible defaults.
    # רק לדמיון יש שדה חובה אמיתי (שם שחקן) — שאר הכוונות לא נחסמות על "חסר"
    if parsed["missing"] and parsed["intent"] == "similar_players":
        return {**base, "status": "clarify",
                "message": ask_clarifying_question(parsed["missing"], parsed["language"])}

    # מנסים להריץ את השאילתה דרך הניתוב
    try:
        result, extra = route_query(parsed["intent"], parsed["filters"], df)
    except ValueError as e:
        # אם השחקן לא נמצא במאגר — הודעה ידידותית
        if "player not found" in str(e):  # named player isn't in the FC24 dataset
            # שם השחקן שחיפשו
            name = parsed["filters"].get("player_name", "")
            # הודעה בעברית
            if parsed["language"] == "he":
                msg = (f"לא מצאתי את '{name}' במאגר FC24 — ייתכן שהשחקן פרש או "
                       f"שהשם נכתב אחרת. נסה שם אחר.")
            else:
                # הודעה באנגלית
                msg = (f"I couldn't find '{name}' in the FC24 dataset — the player "
                       f"may be retired or spelled differently. Try another name.")
            # מחזירים סטטוס "לא נמצא"
            return {**base, "status": "not_found", "message": msg}
        # שגיאת ערך אחרת — מחזירים שגיאה כללית
        return {**base, "status": "error", "message": f"could not run the query: {e}"}
    except Exception as e:
        # כל שגיאה אחרת — מחזירים שגיאה כללית
        return {**base, "status": "error", "message": f"could not run the query: {e}"}

    # הצלחה — מחזירים את התוצאה והמידע הנוסף
    return {**base, "status": "ok", "message": "", "result": result, "extra": extra}


# ===========================================================================
# CONVERSATIONAL AGENT (function-calling) — the natural-language layer.
# The LLM holds a free-flowing conversation, asks guiding questions, CONFIRMS
# before acting, and only then calls one of our Python "tools" (the algorithms).
# The whole chat history is sent every turn, so context is never lost.
# ===========================================================================
SYSTEM_CHAT = """You are "Master Scout", a friendly, sharp AI football-scouting agent.
You ONLY help with scouting INDIVIDUAL football players, using our two datasets:
EA Sports FC24 player profiles + real match-event performance (Football Events).

HOW TO CONVERSE:
- Talk naturally, warmly and BRIEEFLY, in the user's language (Hebrew or English).
  You may answer simple/general questions about yourself and what you can do.
- GUIDED QUESTIONS: if a request is missing useful detail, ask ONE short question
  at a time and build it up. Example — user: "show me a player" → you: "באיזו עמדה?"
  → after they answer → "באיזה טווח גילאים?" → then budget / pace / goals, etc.
  Aim for ~2 criteria, but if the request ALREADY has 2+ details, skip straight to
  the confirmation step — don't ask more. Don't nag.
- POSITIONS: the ONLY positions that exist are Forward, Midfielder, Defender,
  Goalkeeper. NEVER ask for finer roles (center-back vs full-back, winger, striker,
  בלם/מגן) — we don't have that data.
- CONFIRM BEFORE ACTING (exactly once): once you have enough to act (about 2-3
  criteria, and ALWAYS within 3-4 questions total), STOP asking and confirm. In ONE
  clear line: (a) restate what you're about to do (e.g. "אחפש חלוצים בגילאי 25-30 עד
  40 מיליון יורו, ממוין לפי דירוג"), then (b) ask whether to run it as-is OR add more
  detail, AND give 2-4 concrete EXAMPLES of optional criteria the user could still
  add. Tailor the examples to the request and to what's still missing — e.g. for a
  player search: "רוצה שאריץ ככה, או להוסיף עוד קריטריון? אפשר למשל מהירות מינימלית,
  רגל חזקה, ליגה/נבחרת, או ביצועים אמיתיים כמו מינימום גולים/בישולים." (In English:
  "Want me to run it like this, or add a filter? e.g. minimum pace, preferred foot,
  league/nationality, or real stats like minimum goals/assists.") Pick examples that
  FIT the intent (for similar-players: same position only / max age / max budget; for
  bargains: position / minimum rating / max age). Ask this confirmation only ONCE.
  The MOMENT the user agrees ("כן" / "בצע" / "חפש" / "go" / "sure" / "תריץ" / "כמו
  שזה") you MUST call the matching tool immediately — do NOT ask again. If the user
  adds a criterion instead, fold it in and re-confirm once (with examples again only
  if helpful). If the user includes a go-word ("בצע"/"כן"/"חפש"/"go") TOGETHER with
  their last detail, treat it as the confirmation and call the tool right away.
- After a tool runs you'll get its result; summarize it briefly and offer a natural
  next step (e.g. "רוצה לצמצם לפי תקציב? או לראות שחקן דומה?").
- BE EFFICIENT: never ask more than 3-4 questions before reaching the confirmation
  step. By your 3rd-4th question you should already have enough to act — confirm and
  offer optional extra filters rather than interrogating further.

CLUSTERING: NEVER ask the user how many groups (K) to use — the system always picks
the OPTIMAL number automatically (you may mention "אבחר את החלוקה האופטימלית"). You
MAY ask which position to focus on, or whether to use all players.

GOALKEEPERS: goalkeepers have no outfield play-style attributes in our data, so
STATISTICAL analyses (play-style similarity, clustering) DON'T apply to them yet.
For ANY request about a goalkeeper (profile lookup OR similar players), still call
the matching tool with that name — the system automatically returns the goalkeeper's
BASIC details only (name, age, overall, potential) plus a clear notice that fuller
statistical analysis will be added in a future update. Just relay that; never invent
goalkeeper play-style stats or similar-player lists.

DISAMBIGUATION: if a name matches several different players, the tool returns a short
list; present it and ask the user which specific player they mean.

YOU DON'T KNOW OUR DATA: you have NO knowledge of what is or isn't inside our
datasets. For ANY request about a SPECIFIC named player (profile/card, similarity,
their stats) you MUST call a tool — the tool checks our data and falls back to the
domain web by itself. NEVER claim from your own assumption that a player "is not in
our database" without calling a tool first. For "ספר לי / תראה לי על <player>" call
player_profile directly (no confirmation needed for a simple lookup).

DATA SOURCES (priority order) — always our data first, external only as a fallback:
  1. PRIMARY   = our EA FC24 profile dataset.
  2. SECONDARY = our real match-events dataset.
  3. EA OFFICIAL = the live EA Sports FC official ratings API (real, current data +
     a player photo), queried when a named player isn't in our datasets.
  4. MODEL = the model's own football knowledge — only if EA has nothing either.
The tool result tells you which source was used (source = primary / ea / web). END
every data answer with a short source line, e.g. "📊 מבוסס על: מאגר ראשי (FC24)".
When source is "ea": say "השחקן אינו במאגרים שלנו — הנתונים נשלפו מהדירוגים הרשמיים
של EA Sports FC". When source is "web": say "...נבנה בעזרת המודל לפי ידע על השחקן".

SCOPE & ETHICS:
- Politely refuse anything that is NOT about individual players (team/tactical
  analysis, match/score prediction, betting, tickets, general knowledge).
- You RECOMMEND only — never decide to sign/buy a player.
- Never invent players, numbers or stats inside our datasets — the tools provide
  those; the domain-web fallback is clearly labelled as model-generated.

NAMES: ALWAYS pass player names to the tools in ENGLISH — transliterate Hebrew
first (e.g. "לאמין יאמל"→"Lamine Yamal", "מסי"→"Messi", "אוליסה"→"Olise"). Never
pass Hebrew text as a player_name.
"""

# the tools the agent can call — each maps to one of our algorithms
# רשימת הכלים שה-LLM יכול לקרוא להם — כל אחד ממופה לאלגוריתם שלנו
TOOLS = [
    # כלי 1: חיפוש/סינון שחקנים לפי פרופיל וביצועים
    {"type": "function", "function": {
        "name": "search_players",
        "description": ("Search/filter players by profile and real-performance "
                        "criteria. Use for any 'find/show players with...' request "
                        "and for narrowing a previous list (combine all criteria "
                        "mentioned so far)."),
        "parameters": {"type": "object", "properties": {
            "position": {"type": "string", "enum": ["Forward", "Midfielder", "Defender", "GK"]},
            "min_age": {"type": "integer"}, "max_age": {"type": "integer"},
            "min_overall": {"type": "integer"}, "min_potential": {"type": "integer"},
            "min_pace": {"type": "integer"},
            "min_value_eur": {"type": "integer"}, "max_value_eur": {"type": "integer"},
            "min_total_goals": {"type": "integer", "description": "goals / שערים / הבקעות"},
            "min_key_passes": {"type": "integer", "description": "assists≈key passes / בישולים"},
            "min_shots": {"type": "integer"}, "min_braces": {"type": "integer",
                "description": "two-goal games / צמדי גולים"},
            "preferred_foot": {"type": "string", "enum": ["Left", "Right"]},
            "top_n": {"type": "integer"},
        }}}},
    # כלי 2: מציאת שחקנים דומים בסגנון לשחקן נתון (Cosine)
    {"type": "function", "function": {
        "name": "find_similar_players",
        "description": "Find players most similar in play-style to a named player (Cosine).",
        "parameters": {"type": "object", "properties": {
            "player_name": {"type": "string"}, "top_n": {"type": "integer"},
            "same_position": {"type": "boolean"},
            "max_age": {"type": "integer"}, "max_value_eur": {"type": "integer"},
        }, "required": ["player_name"]}}},
    # כלי 3: קיבוץ שחקנים לסגנונות משחק (K-Means)
    {"type": "function", "function": {
        "name": "cluster_players",
        "description": ("Group players into play-styles with K-Means. Optionally "
                        "within one position; K is chosen automatically if omitted."),
        "parameters": {"type": "object", "properties": {
            "position": {"type": "string", "enum": ["Forward", "Midfielder", "Defender"]},
            "n_clusters": {"type": "integer"},
        }}}},
    # כלי 4: זיהוי שחקני מציאות (יכולת גבוהה מול שווי נמוך)
    {"type": "function", "function": {
        "name": "detect_bargains",
        "description": "Find undervalued players — high ability vs low market value (anomaly).",
        "parameters": {"type": "object", "properties": {
            "position": {"type": "string"}, "min_overall": {"type": "integer"},
            "max_age": {"type": "integer"}, "top_n": {"type": "integer"},
        }}}},
    # כלי 5: הצגת כרטיס פרופיל מלא של שחקן בודד
    {"type": "function", "function": {
        "name": "player_profile",
        "description": ("Show one player's full profile card. ALWAYS call this for ANY "
                        "'tell me about / show me / profile of <name>' request — including "
                        "unusual, new, or Hebrew-spelled names (pass the English form). "
                        "It resolves our data first, then EA's official ratings API, then "
                        "model knowledge. NEVER answer such a request, and never say a "
                        "player wasn't found, without calling this tool first."),
        "parameters": {"type": "object", "properties": {
            "player_name": {"type": "string"}}, "required": ["player_name"]}}},
]

# מילון הממפה שם כלי לשם הכוונה המתאימה בניתוב
_TOOL_INTENT = {"search_players": "profile_search", "find_similar_players": "similar_players",
                "cluster_players": "clustering", "detect_bargains": "bargains"}


# פונקציית עזר: מנקה את ארגומנטי הכלי ומתרגמת position ל-position_group
def _tool_filters(args: dict) -> dict:
    # שומרים רק ערכים שאינם None
    f = {k: v for k, v in (args or {}).items() if v is not None}
    # ממפים את שם הפרמטר position לשם הפנימי position_group
    if "position" in f:
        f["position_group"] = f.pop("position")
    # מחזירים את המסננים
    return f


# פונקציה: בונה סיכום טקסטואלי קצר של התוצאה כדי שה-LLM יוכל לתאר אותה
def _summarize_result(name: str, res: pd.DataFrame, extra: dict) -> str:
    """A compact text summary fed back to the LLM so it can narrate the result."""
    # מקרה כרטיס שחקן בודד
    if name == "player_profile":
        # השורה הראשונה
        r = res.iloc[0]
        # סיכום פרטי השחקן
        return (f"{r['short_name']}: position {r.get('position_group')}, age "
                f"{int(r['age'])}, overall {int(r['overall'])}, value €{int(r['value_eur']):,}.")
    # מקרה קיבוץ (קיימת עמודת label)
    if "label" in res.columns:  # clustering
        # מרכיבים תיאור לכל סגנון עם גודלו
        parts = [f"{row['label']} (n={row['size']})" for _, row in res.iterrows()]
        # מספר הקלאסטרים
        k = extra.get("k")
        # סיכום הסגנונות
        return f"K={k} play-styles: " + "; ".join(parts)
    # מקרה רשימת שחקנים (קיימת עמודת short_name)
    if "short_name" in res.columns:
        # 5 השמות הראשונים
        top = ", ".join(str(x) for x in res["short_name"].head(5))
        # סיכום: כמה נמצאו ומי המובילים
        return f"{len(res)} players found. Top: {top}." if len(res) else "No players matched."
    # ברירת מחדל: מספר שורות
    return f"{len(res)} rows."


# ---- external source fallback (source priority: our data -> domain web) --------
# שדות הכרטיס שנבקש מהמקור החיצוני (EA / ידע המודל)
_WEB_FIELDS = ("short_name", "long_name", "club_name", "position_group", "age",
               "overall", "potential", "value_eur", "preferred_foot",
               "nationality_name", "pace", "shooting", "passing", "dribbling",
               "defending", "physic")


# פונקציה: גיבוי ידע-מודל — בונה תכונות בסגנון FC לשחקן שאינו במאגרים (המקור האחרון)
def web_player_lookup(player_name: str) -> dict | None:
    """DOMAIN-WEB fallback: use the model's football knowledge (the wider domain —
    e.g. official EA FC ratings / fifaindex) to build FC-style attributes for a
    player NOT in our datasets. Returns a dict, or None if it's not a real player.
    This is the LAST source in the priority chain (our data first, then this)."""
    # מקבלים את לקוח ה-LLM
    client = _get_client()
    # בונים את הפרומפט שמבקש תכונות בסגנון EA רק אם זה שחקן אמיתי
    prompt = (
        f'For the football player "{player_name}", if and ONLY IF this is a REAL '
        "professional footballer, return their EA-FC-style attributes from your "
        "knowledge (use public ratings such as EA Sports FC / fifaindex as a guide). "
        'Reply with JSON only: {"is_real_player": true/false, "short_name": str, '
        '"long_name": str, "club_name": str, "position_group": one of '
        '"Forward"|"Midfielder"|"Defender"|"GK", "age": int, "overall": int 40-99, '
        '"potential": int, "value_eur": int, "preferred_foot": "Left"|"Right", '
        '"nationality_name": str, "pace": int 0-99, "shooting": int, "passing": int, '
        '"dribbling": int, "defending": int, "physic": int, "note": str}. '
        'If the name is NOT a real footballer, reply {"is_real_player": false}.')
    # מנסים לקרוא ל-LLM ולפענח את ה-JSON
    try:
        # קריאה ל-LLM בפורמט JSON
        resp = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        # מפענחים את התשובה
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        # כל כשל → אין תוצאה
        return None
    # מחזירים את הנתונים רק אם זה שחקן אמיתי
    return data if data.get("is_real_player") else None


# פונקציה: בונה שורת כרטיס בודדת ממילון חיצוני (EA או ידע המודל)
def _ext_card_row(w: dict) -> dict:
    """Build a one-row card from an external dict (EA API or model knowledge)."""
    # שולפים את כל שדות הכרטיס מהמילון החיצוני
    row = {k: w.get(k) for k in _WEB_FIELDS}
    # שם קצר: קצר אם יש, אחרת מלא, אחרת מקף
    row["short_name"] = row.get("short_name") or row.get("long_name") or "—"
    # כתובת תמונה (EA מספק תמונת שחקן)
    row["avatar_url"] = w.get("avatar_url")              # EA gives a player photo
    # כתובת דגל הלאום
    row["nationality_flag_url"] = w.get("nationality_flag_url")
    # כתובת סמל המועדון
    row["club_badge_url"] = w.get("club_badge_url")
    # שחקן חיצוני — אין לו נתוני אירועים
    row["has_event_data"] = False
    # מחזירים את שורת הכרטיס
    return row


# פונקציה: שרשרת המקור החיצוני לשחקן שאינו במאגר (EA → ידע המודל)
def external_lookup(name: str) -> dict | None:
    """External-source chain for a player not in our data:
    1) EA Sports FC official ratings API (real, with photo), then
    2) the model's own football knowledge (last resort)."""
    # ראשית מנסים את ה-API הרשמי של EA
    ea = external.ea_fc_lookup(name)
    # אם נמצא ויש לו דירוג — מחזירים אותו
    if ea and ea.get("overall"):
        return ea
    # אחרת מנסים את ידע המודל
    web = web_player_lookup(name)
    # אם נמצא — מסמנים שהמקור הוא ידע המודל
    if web:
        web["_source"] = "web"
    # מחזירים את התוצאה (או None)
    return web


# פונקציה: מריצה כלי → (טבלת תוצאה, מידע נוסף, סיכום טקסטואלי), עם עדיפות מקור
def run_tool(name: str, args: dict, df: pd.DataFrame):
    """Execute a tool -> (result_df, extra, summary_text).
    Source priority: our two datasets first; only if a NAMED player isn't there do
    we fall back to the domain-web (the model's football knowledge)."""
    # כלי כרטיס פרופיל
    if name == "player_profile":
        # שם השחקן המבוקש
        nm = args.get("player_name", "")
        # מחפשים התאמות שם במאגר
        matches = similarity.find_player_matches(df, nm)
        if len(matches) == 1:                                   # one clear player
            # goalkeeper -> show only basic details + the "coming soon" notice
            # שוער → מחזירים רק פרטים בסיסיים + הודעת "בעתיד"
            if str(matches.iloc[0]["position_group"]) == "GK":
                raise GoalkeeperAnalysisUnavailable(_basic_facts(matches.iloc[0]))
            # שורת התוצאה היחידה
            res = matches.iloc[[0]].reset_index(drop=True)
            # מחזירים כרטיס ממקור ראשי
            return res, {"card": True, "source": "primary"}, _summarize_result(name, res, {})
        if len(matches) > 1:                                    # disambiguate
            # התאמות מרובות → רשימת הבחנה
            cols = [c for c in ("short_name", "long_name", "position_group", "age",
                                "overall", "club_name") if c in matches.columns]
            # טבלת המועמדים
            res = matches[cols].reset_index(drop=True)
            # רשימה טקסטואלית של המועמדים
            lst = "; ".join(f"{r.short_name} ({getattr(r, 'club_name', '')})"
                            for r in matches.itertuples())
            # מחזירים סטטוס הבחנה ומבקשים מהמשתמש לבחור
            return res, {"disambiguation": True, "source": "primary"}, \
                f"Several players match '{nm}': {lst}. Ask the user which one."
        # לא נמצא במאגר → מנסים מקור חיצוני (EA ואז מודל)
        ext = external_lookup(nm)                               # EA API, then model
        # אם גם חיצונית לא נמצא — שגיאה
        if ext is None:
            raise ValueError(f"player not found anywhere: {nm}")
        # שוער חיצוני → פרטים בסיסיים + הודעה
        if str(ext.get("position_group")) == "GK":              # external goalkeeper
            raise GoalkeeperAnalysisUnavailable(_basic_facts(ext))
        # מקור הנתונים (ea/web)
        src = ext.get("_source", "web")
        # בונים שורת כרטיס מהמקור החיצוני
        res = pd.DataFrame([_ext_card_row(ext)])
        # ניסוח מקור הנתונים
        where = ("EA Sports FC official ratings" if src == "ea"
                 else "the model's domain knowledge")
        # מחזירים כרטיס עם ציון המקור
        return res, {"card": True, "source": src,
                     "avatar": ext.get("avatar_url"), "source_url": ext.get("_source_url")}, \
            f"{res.iloc[0]['short_name']} — NOT in our datasets; profile from {where}."

    # כלי שחקנים דומים
    if name == "find_similar_players":
        # שם השחקן המבוקש
        nm = args.get("player_name", "")
        # מאתרים את שורת השחקן במאגר
        row = similarity._find_player_row(df, nm)
        # אם נמצא במאגר
        if row is not None:
            # goalkeeper in our data -> basic details only + "coming soon" notice
            # שוער במאגר → פרטים בסיסיים בלבד + הודעה
            if str(row["position_group"]) == "GK":
                raise GoalkeeperAnalysisUnavailable(_basic_facts(row))
            # מריצים דמיון רגיל
            res, extra = route_query("similar_players", _tool_filters(args), df)
            # מסמנים מקור ראשי
            extra["source"] = "primary"
            # מחזירים תוצאה + סיכום
            return res, extra, _summarize_result(name, res, extra)
        # לא במאגר → מקור חיצוני
        ext = external_lookup(nm)                               # target not in data
        # אם לא נמצא חיצונית — שגיאה
        if ext is None:
            raise ValueError(f"player not found anywhere: {nm}")
        # שוער חיצוני כיעד → פרטים בסיסיים + הודעה
        if str(ext.get("position_group")) == "GK":              # external GK target
            raise GoalkeeperAnalysisUnavailable(_basic_facts(ext))
        # אם התבקשה אותה עמדה — מגבילים אליה
        pos = ext.get("position_group") if args.get("same_position") else None
        # מחשבים דמיון לפי תכונות היעד החיצוני
        res = similarity.find_similar_to_attrs(
            df, ext, top_n=int(args.get("top_n") or 5), position_group=pos)
        # שם היעד ומקורו
        tgt, src = ext.get("short_name", nm), ext.get("_source", "web")
        # ניסוח מקור הנתונים
        where = "EA Sports FC official ratings" if src == "ea" else "the model"
        # carry the target's full attribute row so the UI can draw its radar/card
        # even though it is NOT in our local table.
        # מצרפים את שורת התכונות של היעד כדי ש-UI יוכל לצייר רדאר/כרטיס
        return res, {"target": tgt, "source": src, "target_row": _ext_card_row(ext)}, \
            (f"Target {tgt} is NOT in our data — used its attributes from {where}. "
             f"Closest in our data: " + ", ".join(res["short_name"].head(5).astype(str)))

    # שאר הכלים → ניתוב לפי המיפוי
    intent = _TOOL_INTENT[name]
    # מריצים את הניתוב
    res, extra = route_query(intent, _tool_filters(args), df)
    # מסמנים מקור ראשי
    extra["source"] = "primary"
    # מחזירים תוצאה + סיכום
    return res, extra, _summarize_result(name, res, extra)


# פונקציית עזר: ממירה אובייקט tool_call של OpenAI למילון פשוט
def _tc_to_dict(tc):
    # בונה מילון תואם לפורמט ההיסטוריה של OpenAI
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}


def _user_language(messages: list) -> str:
    """Guess the user's language ('he'/'en') from the latest user message —
    if it contains any Hebrew letter we answer in Hebrew, otherwise English."""
    # עוברים על ההודעות מהאחרונה לראשונה
    for m in reversed(messages):
        # מאתרים את הודעת המשתמש האחרונה עם תוכן
        if m.get("role") == "user" and m.get("content"):
            # אם יש בה אות עברית — עברית, אחרת אנגלית
            return "he" if any("֐" <= ch <= "׿" for ch in m["content"]) else "en"
    # ברירת מחדל: עברית
    return "he"


# פונקציה: תור שיחה אחד — שולחת את ההיסטוריה ל-LLM ומפעילה כלי במידת הצורך
def converse(messages: list, df: pd.DataFrame | None = None):
    """One conversational turn. `messages` is an OpenAI-format history (mutated
    in place with the new assistant/tool turns). Returns (assistant_text, action)
    where action = {"name", "df", "extra"} if a tool ran this turn, else None."""
    # אם לא הועברה טבלה — טוענים את הטבלה המרכזית
    if df is None:
        df = load_table()
    # מקבלים את לקוח ה-LLM
    client = _get_client()
    # קוראים ל-LLM עם הנחיית השיחה, ההיסטוריה והכלים (בחירת כלי אוטומטית)
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_CHAT}] + messages,
        tools=TOOLS, tool_choice="auto")
    # ההודעה שהוחזרה מה-LLM
    msg = resp.choices[0].message

    # אם אין קריאת כלי — זהו תור שיחה רגיל
    if not msg.tool_calls:                       # plain conversation turn
        # מוסיפים את תשובת העוזר להיסטוריה
        messages.append({"role": "assistant", "content": msg.content or ""})
        # מחזירים את הטקסט בלי פעולה
        return msg.content or "", None

    # קריאת הכלי הראשונה
    tc = msg.tool_calls[0]
    # שם הכלי
    name = tc.function.name
    # מנסים לפענח את ארגומנטי הכלי מ-JSON
    try:
        args = json.loads(tc.function.arguments or "{}")
    except Exception:
        # אם נכשל — מילון ריק
        args = {}
    # מוסיפים את הודעת העוזר עם קריאת הכלי להיסטוריה
    messages.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [_tc_to_dict(tc)]})

    # אתחול הפעולה שתוחזר
    action = None
    # מנסים להריץ את הכלי
    try:
        # מריצים את הכלי ומקבלים תוצאה, מידע נוסף וסיכום
        res, extra, summary = run_tool(name, args, df)
        # בונים את אובייקט הפעולה
        action = {"name": name, "df": res, "extra": extra}
    except GoalkeeperAnalysisUnavailable as gk_exc:
        # deliver the goalkeeper notice (basic details + the "coming soon" line)
        # VERBATIM in the user's language and skip the second LLM call, so the exact
        # wording always reaches the user.
        # בונים את הודעת השוער בשפת המשתמש עם הפרטים הבסיסיים
        notice = _gk_notice(_user_language(messages), gk_exc.facts)
        # מוסיפים הודעת tool (חובה אחרי קריאת כלי) עם תקציר פנימי
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": "Goalkeeper: statistical analysis unavailable; "
                                    "delivered the standard notice to the user."})
        # מוסיפים את הודעת השוער כתשובת העוזר
        messages.append({"role": "assistant", "content": notice})
        # מחזירים את ההודעה בלי פעולה (מדלגים על קריאת LLM שנייה)
        return notice, None
    except Exception as e:
        # כל שגיאה אחרת בכלי → סיכום שגיאה
        summary = f"tool error: {e}"
    # מוסיפים את הודעת ה-tool עם הסיכום
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": summary})

    # קריאה שנייה ל-LLM כדי שינסח את התוצאה בשפה טבעית
    resp2 = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": SYSTEM_CHAT}] + messages)
    # הטקסט הסופי
    final = resp2.choices[0].message.content or ""
    # מוסיפים אותו להיסטוריה
    messages.append({"role": "assistant", "content": final})
    # מחזירים את הטקסט הסופי ואת הפעולה (אם רצה כלי)
    return final, action


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# פונקציה: טוענת את הטבלה המרכזית מ-CSV
def load_table(path: Path = DATA_PATH) -> pd.DataFrame:
    # קוראים ומחזירים את הטבלה
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Self-tests / demo
# ---------------------------------------------------------------------------
# בדיקה עצמית של הניתוב — רצה ללא מפתח API (כוונות קשיחות)
def _routing_selftest(df: pd.DataFrame):
    """Exercise the routing plumbing WITHOUT any API key (hardcoded intents)."""
    # כותרת הבדיקה
    print("=" * 70, "\nROUTING SELF-TEST (no API key needed)\n", "=" * 70)
    # מקרי בדיקה: (כוונה, מסננים)
    cases = [
        # חיפוש פרופיל
        ("profile_search", {"position_group": "Forward", "max_age": 23,
                            "min_pace": 85, "max_value_eur": 30_000_000, "top_n": 5}),
        # דמיון
        ("similar_players", {"player_name": "De Bruyne", "top_n": 5}),
        # מציאות
        ("bargains", {"min_overall": 80, "top_n": 5}),
        # דאבלים
        ("braces", {"min_braces": 5, "top_n": 5}),
        # קיבוץ
        ("clustering", {}),
    ]
    # מריצים כל מקרה
    for intent, filters in cases:
        # מנסים להריץ את הניתוב
        try:
            # מריצים את השאילתה
            res, extra = route_query(intent, filters, df)
            # מדפיסים כמה שורות חזרו
            print(f"\n[{intent}] -> {len(res)} rows  filters={filters}")
            # 6 העמודות הראשונות
            cols = [c for c in res.columns][:6]
            # מדפיסים את ראש הטבלה
            print(res[cols].head().to_string(index=False))
            # אם יש מידע נוסף — מדפיסים את מפתחותיו
            if extra:
                print("extra:", list(extra.keys()))
        except Exception as e:
            # כשל — מדפיסים אותו
            print(f"\n[{intent}] FAILED: {e}")


# בדיקה חיה של מסלול ה-GPT — רצה רק אם מוגדר מפתח API
def _live_test(df: pd.DataFrame):
    """Full GPT path — only runs if an API key is configured."""
    # כותרת הבדיקה החיה
    print("\n" + "=" * 70, "\nLIVE AGENT TEST (uses the LLM)\n", "=" * 70)
    # שאילתות הדגמה (חיפוש, דמיון, הבהרה, מציאות, סירוב)
    queries = [
        "מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו, תן 5",
        "who plays like De Bruyne?",
        "תמצא לי שחקנים דומים",                    # should ask to clarify (no name)
        "find undervalued players rated over 80",
        "כמה כרטיסים צהובים קיבלה ריאל מדריד?",      # out of scope (team)
    ]
    # מריצים כל שאילתה דרך הסוכן
    for q in queries:
        # מריצים את הסוכן
        out = run_agent(q, df)
        # מדפיסים את הסטטוס, הכוונה והשפה
        print(f"\nQ: {q}\n  -> status={out['status']} intent={out['intent']} lang={out['language']}")
        # אם הצליח — מדפיסים מספר תוצאות ודוגמה
        if out["status"] == "ok":
            print(f"     {len(out['result'])} results, e.g. "
                  f"{out['result'].iloc[0].get('short_name', '?') if len(out['result']) else '—'}")
        else:
            # אחרת — מדפיסים את ההודעה
            print(f"     {out['message']}")


# בלוק שמורץ בהרצה ישירה — מריץ בדיקת ניתוב ובדיקה חיה אם יש מפתח
def main():
    # טוענים את הטבלה
    df = load_table()
    # מריצים את בדיקת הניתוב (ללא מפתח)
    _routing_selftest(df)
    # אם יש מפתח API — מריצים גם את הבדיקה החיה
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "PASTE_YOUR_KEY_HERE":
        _live_test(df)
    else:
        # אחרת — מדלגים על הבדיקה החיה
        print("\n[no OPENAI_API_KEY yet — skipping the live GPT test. "
              "Paste your key into .env to enable it.]")


# נקודת כניסה
if __name__ == "__main__":
    main()
