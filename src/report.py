"""
report.py — Stage 13. Natural-language scouting reports (NLG).

Takes the structured result from agent.run_agent() and asks the LLM to write a
SHORT, reasoned scouting note in the language of the user's query.

Lecturer's rules baked in: an agent answers SHORT and focused (3-4 players, a
sentence or two of reasoning) — not a Wikipedia entry. The report must state the
DATA SOURCE (FC24 profile only, or also real match events) and connect directly
to what was asked. No filler ("thank you", "great question").

The LLM only PHRASES results it is given — it never invents numbers. All data
comes from our Python tables; this layer just turns rows into prose.
"""

from __future__ import annotations

import os

import pandas as pd

# reuse the agent's provider-isolated client (single place to swap LLM)
try:
    from . import ms_agent as agent
except ImportError:
    import ms_agent as agent

# Reports are the text the user (and the lecturer) actually read, so we use a
# slightly stronger model here than the cheap classifier — still cheap, but it
# stops inventing numbers / leaking template labels. Classification stays on nano.
REPORT_MODEL = os.getenv("OPENAI_REPORT_MODEL", "gpt-4.1-mini")

# intents whose numbers come from REAL match events (vs FC24 profile only)
_EVENT_INTENTS = {
    "attacking_players", "creative_midfielders", "disciplined_defenders",
    "two_footed", "braces", "profile_performance_anomaly",
}


def _source_note(intent: str, language: str) -> str:
    event = intent in _EVENT_INTENTS
    if language == "he":
        return ("נתוני ביצועים אמיתיים ממשחקים (Football Events) יחד עם פרופיל FC24"
                if event else "פרופיל השחקנים של EA Sports FC24")
    return ("real match-event performance (Football Events) combined with the FC24 profile"
            if event else "the EA Sports FC24 player profile")


REPORT_SYSTEM = """You are "Master Scout", writing the final scouting note for the user.

HARD RULES:
- Reply ONLY in the requested language (Hebrew or English). Never echo field
  labels like "DATA_SOURCE" or "INTENT" — write natural prose.
- Keep all PLAYER and CLUB names in their original English spelling exactly as in
  the results, even when the rest of the sentence is Hebrew (e.g. write "M. Salah",
  not "מ. סלאח").
- SHORT: at most 3-4 players, with one short reasoning sentence. You are an agent,
  not Wikipedia. No intro, no "thank you", no "great question".
- When you list several players or clusters, format them as a clean MARKDOWN
  BULLET or NUMBERED list (one item per line, e.g. "- **Name** — reason"), never
  as one long run-on sentence.
- Use ONLY the rows and COLUMNS you are given. NEVER invent a number, club, age,
  score or stat. If a column is not present (e.g. there is no "similarity"
  column), do NOT mention that concept at all.
- Mention the data source naturally in one short clause.
- If a "similarity" column exists: cite its value and the "reason".
- If a "market_efficiency_score" column exists (bargains): say briefly it's a lot
  of ability for a low value. Do NOT mention similarity here.
- If a "direction" column exists (anomaly): use it to explain the mismatch.
- If "label" + "trait_1..5" + "player_1..5" columns exist (clustering): write a
  numbered/bulleted list, ONE line per style — its label, 1-2 example players
  (from player_1..5), and cite the dominant attributes (from trait_1..5) as the
  reason. Each cluster must read differently. List ALL the clusters (one line each).
- Finish with a SHORT note that this is a recommendation only — the agent stops
  before action (it does not sign/buy the player).
- Plain text, no markdown headers."""


def generate_report(result: dict, max_results: int = 4) -> str:
    """Turn an agent.run_agent() result dict into a user-facing report string.

    For non-"ok" statuses (refusal / clarify / error) the message is already
    user-facing, so we return it as-is.
    """
    if result.get("status") != "ok":
        return result.get("message", "")

    intent = result["intent"]
    language = result.get("language", "he")
    df: pd.DataFrame = result["result"].head(max_results)
    source = _source_note(intent, language)

    extra = result.get("extra", {}) or {}
    lang_word = "Hebrew" if language == "he" else "English"
    target_line = (f"These are players whose play-style is closest to {extra['target']}.\n"
                   if extra.get("target") else "")

    user_msg = (
        f"Answer in {lang_word}.\n"
        f"The user asked: {result['query']}\n"
        f"{target_line}"
        f"Mention this data source in one short clause: {source}.\n\n"
        f"Results our models already computed (phrase these exactly — the columns "
        f"below are the ONLY facts you may use):\n"
        f"{df.to_string(index=False)}"
    )

    client = agent._get_client()
    resp = client.chat.completions.create(
        model=REPORT_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


def respond(query: str, df: pd.DataFrame | None = None) -> str:
    """Full pipeline: free text -> agent -> NL report. The one call app.py needs."""
    result = agent.run_agent(query, df)
    return generate_report(result)


def main():
    df = agent.load_table()
    queries = [
        "מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו",
        "who plays like De Bruyne?",
        "find undervalued defenders rated over 80",
        "תמצא לי שחקנים דומים",                 # clarify (no name)
        "כמה כרטיסים צהובים קיבלה ריאל מדריד?",   # refusal
    ]
    for q in queries:
        print("=" * 70)
        print("USER:", q)
        print("-" * 70)
        print(respond(q, df))
        print()


if __name__ == "__main__":
    main()
