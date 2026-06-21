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

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# --- sibling modules: works both as `python src/agent.py` and as `from src.agent ...`
try:
    from . import search, similarity, clustering, anomaly, workingset, external
except ImportError:  # run as a plain script
    import search, similarity, clustering, anomaly, workingset, external

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "final_scouting_table.csv"

# load .env if present (key never lives in the code)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# the model name is a single config knob (env override -> easy to swap)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# the conversation layer (guided questions, confirmation, tool calls) needs strong
# instruction-following, so it uses a more capable model than the legacy classifier.
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

DEFAULT_TOP_N = 4  # lecturer: an agent returns 3-4 results, not a wikipedia dump

# intents the agent can route to (name -> short human description, for docs/tests)
INTENTS = {
    "profile_search": "filter by FC24 profile (position, age, overall, pace, value, foot, league, nationality)",
    "attacking_players": "attackers ranked by real attacking output (event data)",
    "creative_midfielders": "midfielders ranked by chance creation (event data)",
    "disciplined_defenders": "defenders ranked by low-card discipline (event data)",
    "two_footed": "players who shoot well with both feet (event data)",
    "braces": "players with at least N two-goal games (event data)",
    "similar_players": "players similar in play-style to a NAMED player (Cosine)",
    "clustering": "group players into play-styles / describe the styles (K-Means), optionally within one position",
    "bargains": "underpriced players: high ability vs low value (anomaly)",
    "profile_performance_anomaly": "players whose FC24 rating disagrees with real output (anomaly)",
    "visualize": "chart the MOST RECENT results (a follow-up, not a new search)",
    "greeting": "the user is only greeting / making small talk (hi, hello, מה קורה)",
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
_client = None


def _get_client():
    """Lazily build the OpenAI client. Key from env (.env) or Streamlit secrets."""
    global _client
    if _client is not None:
        return _client

    key = os.getenv("OPENAI_API_KEY")
    if not key or key == "PASTE_YOUR_KEY_HERE":
        try:  # on Streamlit Cloud the key lives in st.secrets
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Paste your key into the .env file "
            "(OPENAI_API_KEY=sk-...), or add it to Streamlit secrets."
        )

    from openai import OpenAI
    _client = OpenAI(api_key=key)
    return _client


def _context_block(context: dict | None) -> str:
    """Render the previous command so the LLM can resolve follow-ups."""
    if not context:
        return ""
    prev_intent = context.get("prev_intent")
    if not prev_intent:
        return ""
    prev_filters = json.dumps(context.get("prev_filters") or {}, ensure_ascii=False)
    return (f"CONVERSATION_CONTEXT (for follow-ups only):\n"
            f"PREVIOUS_COMMAND: intent={prev_intent} filters={prev_filters}\n\n"
            f"USER MESSAGE:\n")


def _classify_raw(query: str, context: dict | None = None) -> dict:
    """Single LLM call: free text -> intent/filters JSON. Returns a raw dict.

    `context` (optional) carries the previous command so the model can resolve
    follow-ups like "expand to 10" or "visualize that".
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _context_block(context) + query},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# Public agent API
# ---------------------------------------------------------------------------
def classify_and_extract(query: str, context: dict | None = None) -> dict:
    """Call the LLM and normalize the result to a guaranteed schema."""
    raw = _classify_raw(query, context)
    return {
        "in_scope": bool(raw.get("in_scope", False)),
        "intent": raw.get("intent"),
        "filters": dict(raw.get("filters") or {}),
        "missing": list(raw.get("missing") or []),
        "language": raw.get("language", "he"),
        "refusal": raw.get("refusal", ""),
    }


_CLARIFY_FIELDS_HE = {
    "player_name": "שם השחקן להשוואה",
    "min_braces": "כמה משחקי דאבל (2+ גולים) לפחות",
    "position_group": "עמדה (חלוץ/קשר/הגנה)",
}
_CLARIFY_FIELDS_EN = {
    "player_name": "the player's name to compare against",
    "min_braces": "the minimum number of two-goal games",
    "position_group": "a position (Forward/Midfielder/Defender)",
}


def ask_clarifying_question(missing: list, language: str = "he") -> str:
    """Turn a list of missing fields into one short clarifying question."""
    if language == "he":
        parts = [_CLARIFY_FIELDS_HE.get(m, m) for m in missing]
        return "כדי להמשיך אני צריך עוד פרט: " + ", ".join(parts) + ". תוכל להוסיף?"
    parts = [_CLARIFY_FIELDS_EN.get(m, m) for m in missing]
    return "To continue I need a bit more: " + ", ".join(parts) + ". Could you add that?"


def _default_refusal(language: str = "he") -> str:
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
    if language == "he":
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
def _pick(filters: dict, keys) -> dict:
    """Keep only the allowed keys for a given backend function."""
    return {k: filters[k] for k in keys if k in filters and filters[k] is not None}


_PROFILE_KEYS = ("position_group", "max_age", "min_age", "min_overall",
                 "min_potential", "min_pace", "max_value_eur", "min_value_eur",
                 "preferred_foot", "league_name", "nationality")


def route_query(intent: str, filters: dict, df: pd.DataFrame):
    """Dispatch an intent+filters to the matching backend function.

    Returns (result_df, extra) where extra is a dict with anything beyond the
    main table (e.g. cluster descriptions, the similarity target name).
    Raises ValueError on an unknown intent or a bad parameter (e.g. player not found).
    """
    f = dict(filters or {})
    top_n = int(f.get("top_n", DEFAULT_TOP_N))
    extra = {}

    # all the filter/search intents go through ONE generic engine so that ANY
    # recognized filter works (free-text robustness) and display == working set.
    if intent in ("profile_search", "attacking_players", "creative_midfielders",
                  "disciplined_defenders", "two_footed", "braces"):
        res, _full = workingset.search(df, intent, f, top_n=top_n)

    elif intent == "similar_players":
        name = f.get("player_name")
        if not name:
            raise ValueError("similar_players requires player_name")
        res, target = similarity.find_similar_players(
            df, name, top_n=top_n,
            **_pick(f, ("same_position", "max_age", "max_value_eur")))
        extra["target"] = str(target.get("short_name", name))

    elif intent == "clustering":
        # per-position clustering: if a position is named, cluster only within it
        pos = f.get("position_group")
        df_c = df[df["position_group"] == pos].copy() if pos else df
        # let the algorithm pick the optimal K (3-6) unless the user named one
        if f.get("n_clusters"):
            n_clusters, sil = int(f["n_clusters"]), None
        else:
            n_clusters, sil = clustering.best_k(df_c)
        labeled, _model, _scaler = clustering.run_player_kmeans(df_c, n_clusters=n_clusters)
        ids = sorted(c for c in labeled["cluster_id"].unique() if c != -1)
        descriptions = clustering.describe_clusters(labeled, ids)  # distinct labels
        # one value per cell: split the dominant traits and sample players into
        # their own columns (trait_1..5, player_1..5)
        rows = []
        for d in descriptions:
            row = {"cluster_id": d["cluster_id"], "label": d["label"], "size": d["size"]}
            for i, t in enumerate(d["dominant_traits"][:5], 1):
                row[f"trait_{i}"] = f"{t['trait']} {t['mean']:.0f}"
            for i, name in enumerate(d["sample_players"][:5], 1):
                row[f"player_{i}"] = name
            rows.append(row)
        res = pd.DataFrame(rows)
        extra["descriptions"] = descriptions
        extra["labeled"] = labeled            # full rows + cluster_id, for drilling
        extra["k"] = n_clusters
        extra["auto_k"] = sil is not None     # True if the algorithm chose K
        extra["silhouette"] = sil
        if pos:
            extra["position"] = pos

    elif intent == "bargains":
        res = anomaly.detect_bargain_players(
            df, top_n=top_n, **_pick(f, ("contamination", "min_overall",
                                         "max_age", "max_value_eur")))

    elif intent == "profile_performance_anomaly":
        res = anomaly.detect_profile_performance_anomalies(
            df, top_n=top_n, **_pick(f, ("contamination",)))

    else:
        raise ValueError(f"unknown intent: {intent!r}")

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
    if df is None:
        df = load_table()
    try:
        parsed = classify_and_extract(query, context)
    except RuntimeError as e:  # no API key
        return {"query": query, "intent": None, "filters": {}, "language": "he",
                "result": None, "extra": {}, "status": "error", "message": str(e)}
    except Exception as e:  # network / parsing
        return {"query": query, "intent": None, "filters": {}, "language": "he",
                "result": None, "extra": {}, "status": "error",
                "message": f"agent error: {e}"}
    return execute_parsed(parsed, df, query)


def execute_parsed(parsed: dict, df: pd.DataFrame, query: str = "") -> dict:
    """Run an already-classified command (lets the caller inspect `parsed` first,
    e.g. to ask a guided question, without re-calling the LLM)."""
    base = {"query": query, "intent": None, "filters": {}, "language": "he",
            "result": None, "extra": {}}
    base.update(intent=parsed["intent"], filters=parsed["filters"],
                language=parsed["language"])

    # "visualize" is a follow-up handled by the caller (it charts the last result)
    if parsed["intent"] == "visualize":
        return {**base, "status": "visualize", "message": ""}

    # a plain greeting -> warm intro (never refuse small talk)
    if parsed["intent"] == "greeting":
        return {**base, "status": "greeting",
                "message": greeting_message(parsed["language"])}

    # reset -> the caller clears the working set
    if parsed["intent"] == "reset":
        return {**base, "status": "reset", "message": ""}

    if not parsed["in_scope"]:
        return {**base, "status": "out_of_scope",
                "message": parsed["refusal"] or _default_refusal(parsed["language"])}

    # only similar_players has a truly required field (player_name); never block
    # other intents on "missing" — they all have sensible defaults.
    if parsed["missing"] and parsed["intent"] == "similar_players":
        return {**base, "status": "clarify",
                "message": ask_clarifying_question(parsed["missing"], parsed["language"])}

    try:
        result, extra = route_query(parsed["intent"], parsed["filters"], df)
    except ValueError as e:
        if "player not found" in str(e):  # named player isn't in the FC24 dataset
            name = parsed["filters"].get("player_name", "")
            if parsed["language"] == "he":
                msg = (f"לא מצאתי את '{name}' במאגר FC24 — ייתכן שהשחקן פרש או "
                       f"שהשם נכתב אחרת. נסה שם אחר.")
            else:
                msg = (f"I couldn't find '{name}' in the FC24 dataset — the player "
                       f"may be retired or spelled differently. Try another name.")
            return {**base, "status": "not_found", "message": msg}
        return {**base, "status": "error", "message": f"could not run the query: {e}"}
    except Exception as e:
        return {**base, "status": "error", "message": f"could not run the query: {e}"}

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
- CONFIRM BEFORE ACTING (exactly once): when you have ~2+ criteria, restate in ONE
  clear line what you're about to do (e.g. "אחפש חלוצים בגילאי 25-30 עד 40 מיליון
  יורו, ממוין לפי דירוג") and ask to confirm or edit ("לבצע? או לשנות משהו?").
  This single confirmation question is separate from gathering criteria — ask it
  only ONCE. The MOMENT the user agrees ("כן" / "בצע" / "חפש" / "go" / "sure" /
  "תריץ") you MUST call the matching tool immediately — do NOT ask again. If the
  user asks to change something, adjust and re-confirm once. If the user includes a
  go-word ("בצע"/"כן"/"חפש"/"go") TOGETHER with their last detail, treat it as the
  confirmation and call the tool right away — do NOT add another confirmation step.
- After a tool runs you'll get its result; summarize it briefly and offer a natural
  next step (e.g. "רוצה לצמצם לפי תקציב? או לראות שחקן דומה?").
- BE EFFICIENT: reach an actionable result within about 3-4 of your questions —
  don't over-interrogate. After ~2 criteria, go to the confirmation step.

CLUSTERING: NEVER ask the user how many groups (K) to use — the system always picks
the OPTIMAL number automatically (you may mention "אבחר את החלוקה האופטימלית"). You
MAY ask which position to focus on, or whether to use all players.

GOALKEEPERS: goalkeepers have no outfield play-style attributes in our data, so
clustering and play-style similarity DON'T apply to them — say so if asked.

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
TOOLS = [
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
    {"type": "function", "function": {
        "name": "find_similar_players",
        "description": "Find players most similar in play-style to a named player (Cosine).",
        "parameters": {"type": "object", "properties": {
            "player_name": {"type": "string"}, "top_n": {"type": "integer"},
            "same_position": {"type": "boolean"},
            "max_age": {"type": "integer"}, "max_value_eur": {"type": "integer"},
        }, "required": ["player_name"]}}},
    {"type": "function", "function": {
        "name": "cluster_players",
        "description": ("Group players into play-styles with K-Means. Optionally "
                        "within one position; K is chosen automatically if omitted."),
        "parameters": {"type": "object", "properties": {
            "position": {"type": "string", "enum": ["Forward", "Midfielder", "Defender"]},
            "n_clusters": {"type": "integer"},
        }}}},
    {"type": "function", "function": {
        "name": "detect_bargains",
        "description": "Find undervalued players — high ability vs low market value (anomaly).",
        "parameters": {"type": "object", "properties": {
            "position": {"type": "string"}, "min_overall": {"type": "integer"},
            "max_age": {"type": "integer"}, "top_n": {"type": "integer"},
        }}}},
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

_TOOL_INTENT = {"search_players": "profile_search", "find_similar_players": "similar_players",
                "cluster_players": "clustering", "detect_bargains": "bargains"}


def _tool_filters(args: dict) -> dict:
    f = {k: v for k, v in (args or {}).items() if v is not None}
    if "position" in f:
        f["position_group"] = f.pop("position")
    return f


def _summarize_result(name: str, res: pd.DataFrame, extra: dict) -> str:
    """A compact text summary fed back to the LLM so it can narrate the result."""
    if name == "player_profile":
        r = res.iloc[0]
        return (f"{r['short_name']}: position {r.get('position_group')}, age "
                f"{int(r['age'])}, overall {int(r['overall'])}, value €{int(r['value_eur']):,}.")
    if "label" in res.columns:  # clustering
        parts = [f"{row['label']} (n={row['size']})" for _, row in res.iterrows()]
        k = extra.get("k")
        return f"K={k} play-styles: " + "; ".join(parts)
    if "short_name" in res.columns:
        top = ", ".join(str(x) for x in res["short_name"].head(5))
        return f"{len(res)} players found. Top: {top}." if len(res) else "No players matched."
    return f"{len(res)} rows."


# ---- external source fallback (source priority: our data -> domain web) --------
_WEB_FIELDS = ("short_name", "long_name", "club_name", "position_group", "age",
               "overall", "potential", "value_eur", "preferred_foot",
               "nationality_name", "pace", "shooting", "passing", "dribbling",
               "defending", "physic")


def web_player_lookup(player_name: str) -> dict | None:
    """DOMAIN-WEB fallback: use the model's football knowledge (the wider domain —
    e.g. official EA FC ratings / fifaindex) to build FC-style attributes for a
    player NOT in our datasets. Returns a dict, or None if it's not a real player.
    This is the LAST source in the priority chain (our data first, then this)."""
    client = _get_client()
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
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        return None
    return data if data.get("is_real_player") else None


def _ext_card_row(w: dict) -> dict:
    """Build a one-row card from an external dict (EA API or model knowledge)."""
    row = {k: w.get(k) for k in _WEB_FIELDS}
    row["short_name"] = row.get("short_name") or row.get("long_name") or "—"
    row["avatar_url"] = w.get("avatar_url")              # EA gives a player photo
    row["nationality_flag_url"] = w.get("nationality_flag_url")
    row["club_badge_url"] = w.get("club_badge_url")
    row["has_event_data"] = False
    return row


def external_lookup(name: str) -> dict | None:
    """External-source chain for a player not in our data:
    1) EA Sports FC official ratings API (real, with photo), then
    2) the model's own football knowledge (last resort)."""
    ea = external.ea_fc_lookup(name)
    if ea and ea.get("overall"):
        return ea
    web = web_player_lookup(name)
    if web:
        web["_source"] = "web"
    return web


def run_tool(name: str, args: dict, df: pd.DataFrame):
    """Execute a tool -> (result_df, extra, summary_text).
    Source priority: our two datasets first; only if a NAMED player isn't there do
    we fall back to the domain-web (the model's football knowledge)."""
    if name == "player_profile":
        nm = args.get("player_name", "")
        matches = similarity.find_player_matches(df, nm)
        if len(matches) == 1:                                   # one clear player
            res = matches.iloc[[0]].reset_index(drop=True)
            return res, {"card": True, "source": "primary"}, _summarize_result(name, res, {})
        if len(matches) > 1:                                    # disambiguate
            cols = [c for c in ("short_name", "long_name", "position_group", "age",
                                "overall", "club_name") if c in matches.columns]
            res = matches[cols].reset_index(drop=True)
            lst = "; ".join(f"{r.short_name} ({getattr(r, 'club_name', '')})"
                            for r in matches.itertuples())
            return res, {"disambiguation": True, "source": "primary"}, \
                f"Several players match '{nm}': {lst}. Ask the user which one."
        ext = external_lookup(nm)                               # EA API, then model
        if ext is None:
            raise ValueError(f"player not found anywhere: {nm}")
        src = ext.get("_source", "web")
        res = pd.DataFrame([_ext_card_row(ext)])
        where = ("EA Sports FC official ratings" if src == "ea"
                 else "the model's domain knowledge")
        return res, {"card": True, "source": src,
                     "avatar": ext.get("avatar_url"), "source_url": ext.get("_source_url")}, \
            f"{res.iloc[0]['short_name']} — NOT in our datasets; profile from {where}."

    if name == "find_similar_players":
        nm = args.get("player_name", "")
        if similarity._find_player_row(df, nm) is not None:
            res, extra = route_query("similar_players", _tool_filters(args), df)
            extra["source"] = "primary"
            return res, extra, _summarize_result(name, res, extra)
        ext = external_lookup(nm)                               # target not in data
        if ext is None:
            raise ValueError(f"player not found anywhere: {nm}")
        pos = ext.get("position_group") if args.get("same_position") else None
        res = similarity.find_similar_to_attrs(
            df, ext, top_n=int(args.get("top_n") or 5), position_group=pos)
        tgt, src = ext.get("short_name", nm), ext.get("_source", "web")
        where = "EA Sports FC official ratings" if src == "ea" else "the model"
        return res, {"target": tgt, "source": src}, \
            (f"Target {tgt} is NOT in our data — used its attributes from {where}. "
             f"Closest in our data: " + ", ".join(res["short_name"].head(5).astype(str)))

    intent = _TOOL_INTENT[name]
    res, extra = route_query(intent, _tool_filters(args), df)
    extra["source"] = "primary"
    return res, extra, _summarize_result(name, res, extra)


def _tc_to_dict(tc):
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}


def converse(messages: list, df: pd.DataFrame | None = None):
    """One conversational turn. `messages` is an OpenAI-format history (mutated
    in place with the new assistant/tool turns). Returns (assistant_text, action)
    where action = {"name", "df", "extra"} if a tool ran this turn, else None."""
    if df is None:
        df = load_table()
    client = _get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM_CHAT}] + messages,
        tools=TOOLS, tool_choice="auto")
    msg = resp.choices[0].message

    if not msg.tool_calls:                       # plain conversation turn
        messages.append({"role": "assistant", "content": msg.content or ""})
        return msg.content or "", None

    tc = msg.tool_calls[0]
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except Exception:
        args = {}
    messages.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [_tc_to_dict(tc)]})

    action = None
    try:
        res, extra, summary = run_tool(name, args, df)
        action = {"name": name, "df": res, "extra": extra}
    except Exception as e:
        summary = f"tool error: {e}"
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": summary})

    resp2 = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3,
        messages=[{"role": "system", "content": SYSTEM_CHAT}] + messages)
    final = resp2.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final})
    return final, action


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_table(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Self-tests / demo
# ---------------------------------------------------------------------------
def _routing_selftest(df: pd.DataFrame):
    """Exercise the routing plumbing WITHOUT any API key (hardcoded intents)."""
    print("=" * 70, "\nROUTING SELF-TEST (no API key needed)\n", "=" * 70)
    cases = [
        ("profile_search", {"position_group": "Forward", "max_age": 23,
                            "min_pace": 85, "max_value_eur": 30_000_000, "top_n": 5}),
        ("similar_players", {"player_name": "De Bruyne", "top_n": 5}),
        ("bargains", {"min_overall": 80, "top_n": 5}),
        ("braces", {"min_braces": 5, "top_n": 5}),
        ("clustering", {}),
    ]
    for intent, filters in cases:
        try:
            res, extra = route_query(intent, filters, df)
            print(f"\n[{intent}] -> {len(res)} rows  filters={filters}")
            cols = [c for c in res.columns][:6]
            print(res[cols].head().to_string(index=False))
            if extra:
                print("extra:", list(extra.keys()))
        except Exception as e:
            print(f"\n[{intent}] FAILED: {e}")


def _live_test(df: pd.DataFrame):
    """Full GPT path — only runs if an API key is configured."""
    print("\n" + "=" * 70, "\nLIVE AGENT TEST (uses the LLM)\n", "=" * 70)
    queries = [
        "מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו, תן 5",
        "who plays like De Bruyne?",
        "תמצא לי שחקנים דומים",                    # should ask to clarify (no name)
        "find undervalued players rated over 80",
        "כמה כרטיסים צהובים קיבלה ריאל מדריד?",      # out of scope (team)
    ]
    for q in queries:
        out = run_agent(q, df)
        print(f"\nQ: {q}\n  -> status={out['status']} intent={out['intent']} lang={out['language']}")
        if out["status"] == "ok":
            print(f"     {len(out['result'])} results, e.g. "
                  f"{out['result'].iloc[0].get('short_name', '?') if len(out['result']) else '—'}")
        else:
            print(f"     {out['message']}")


def main():
    df = load_table()
    _routing_selftest(df)
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "PASTE_YOUR_KEY_HERE":
        _live_test(df)
    else:
        print("\n[no OPENAI_API_KEY yet — skipping the live GPT test. "
              "Paste your key into .env to enable it.]")


if __name__ == "__main__":
    main()
