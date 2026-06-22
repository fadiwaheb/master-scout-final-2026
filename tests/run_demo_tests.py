"""
run_demo_tests.py — Stage 14. Tests, demo cases, and the DOCUMENTED FAILURE case.

Runs a set of end-to-end scenarios through the full agent pipeline
(free text -> classify -> route -> NL report) and writes a markdown evidence
file to reports/14_test_results.md — ready to paste into rubric chapter 7.

Scenario kinds:
  valid      -> should classify to one of `expect_intent` and return results
  refusal    -> should be refused (out of scope) — scope boundary works
  clarify    -> should ask a clarifying question (missing required info)
  not_found  -> named player isn't in the dataset — graceful message
  failure    -> a DELIBERATE failure: the agent misreads the query. We document
                WHAT it does and WHY it is wrong (a real architecture limitation).

Run:  /Users/ronbiton/opt/anaconda3/envs/masterscout/bin/python tests/run_demo_tests.py
"""

# מאפשר תחביר טיפוסים מודרני
from __future__ import annotations

# ייבוא sys להוספת src לנתיב
import sys
# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path

# שורש הפרויקט
ROOT = Path(__file__).resolve().parent.parent
# מוסיפים את src לנתיב הייבוא
sys.path.insert(0, str(ROOT / "src"))

# מייבאים את שכבת הסוכן
import ms_agent as agent      # noqa: E402
# מייבאים את שכבת הדוחות
import report     # noqa: E402

# נתיב קובץ תוצאות הבדיקות (Markdown)
OUT_MD = ROOT / "reports" / "14_test_results.md"


# רשימת תרחישי הבדיקה: תקינים, סירוב, הבהרה, לא-נמצא, וכשלים מתועדים
SCENARIOS = [
    # ---- valid cases (the agent should handle these well) ----
    {
        "id": "V1", "kind": "valid", "expect_intent": {"profile_search"},
        "title": "Profile search (FC24 only)",
        "query": "מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו, תן 3",
    },
    {
        "id": "V2", "kind": "valid",
        "expect_intent": {"attacking_players", "braces", "creative_midfielders"},
        "title": "Performance search (real match events)",
        "query": "מי החלוצים עם הכי הרבה משחקי דאבל (2+ גולים)?",
    },
    {
        "id": "V3", "kind": "valid", "expect_intent": {"similar_players"},
        "title": "Similarity (Cosine) — complex, profile + play-style",
        "query": "who plays like De Bruyne and is worth under 40M?",
    },
    {
        "id": "V4", "kind": "valid", "expect_intent": {"bargains"},
        "title": "Bargains (anomaly) — high ability vs low value",
        "query": "find undervalued defenders rated over 80",
    },
    # ---- scope boundary (the agent should refuse) ----
    {
        "id": "S1", "kind": "refusal",
        "title": "Out of scope — team question",
        "query": "כמה כרטיסים צהובים קיבלה ריאל מדריד העונה?",
    },
    {
        "id": "S2", "kind": "refusal",
        "title": "Out of scope — real-world action (stop-before-action)",
        "query": "תחתים לי את Haaland לקבוצה שלי",
    },
    # ---- clarifying question (missing required info) ----
    {
        "id": "C1", "kind": "clarify",
        "title": "Clarify — similarity without a player name",
        "query": "תמצא לי שחקנים דומים",
    },
    # ---- graceful not-found (data coverage limit) ----
    {
        "id": "N1", "kind": "not_found",
        "title": "Player not in FC24 dataset (retired) — graceful handling",
        "query": "שחקנים דומים לרונאלדיניו",
    },
    # ---- THE DOCUMENTED FAILURE CASE (rubric ch.7, scored) ----
    {
        "id": "F1", "kind": "failure",
        "title": "FAILURE — mixed-intent query the single-intent router can't satisfy",
        "query": "מצא חלוצים מהירים שדומים למסי",
        "why": (
            "The user asks for TWO things at once: a profile filter ('fast forwards') "
            "AND a similarity target ('like Messi'). Our agent routes each message to "
            "exactly ONE intent, so it picks similar_players(Messi) and SILENTLY DROPS "
            "the 'fast forwards' constraints — returning players similar to Messi with no "
            "guarantee they are fast forwards. Root cause: single-intent, one-shot routing. "
            "Fix direction: a multi-step agent loop (Plan->Act) or a post-similarity profile "
            "filter — listed as a future improvement, not in the MVP scope."
        ),
    },
    {
        "id": "F2", "kind": "failure",
        "title": "FAILURE — false refusal triggered by the word 'קבוצה' (group/team)",
        "query": "תן לי קבוצה של חלוצים טובים",
        "why": (
            "The user means a GROUP of forwards, but the word 'קבוצה' (which also means "
            "'team') trips the scope boundary and the agent wrongly refuses as a team "
            "question. Root cause: keyword-sensitive scope detection over-refuses. "
            "Fix direction: rely on the LLM's intent rather than surface keywords, or add "
            "'group of players' as an explicit in-scope example in the system prompt."
        ),
    },
]


# פונקציה: מריצה תרחיש בודד דרך הסוכן ומחזירה (פלט, תשובה, פסק-דין)
def _run_one(sc, df):
    # מריצים את השאילתה דרך הסוכן
    out = agent.run_agent(sc["query"], df)
    # build the user-facing answer (report for ok, else the status message)
    # התשובה למשתמש: דוח אם הצליח, אחרת הודעת הסטטוס
    answer = report.generate_report(out) if out["status"] == "ok" else out["message"]

    # decide PASS/observed for the summary
    # קובעים פסק-דין לפי סוג התרחיש
    if sc["kind"] == "valid":
        # תקין: עבר אם הסטטוס ok והכוונה צפויה
        ok = out["status"] == "ok" and out["intent"] in sc["expect_intent"]
        verdict = "PASS" if ok else "CHECK"
    elif sc["kind"] == "refusal":
        # סירוב: עבר אם הסטטוס מחוץ-לתחום
        verdict = "PASS" if out["status"] == "out_of_scope" else "CHECK"
    elif sc["kind"] == "clarify":
        # הבהרה: עבר אם הסטטוס clarify
        verdict = "PASS" if out["status"] == "clarify" else "CHECK"
    elif sc["kind"] == "not_found":
        # לא-נמצא: עבר אם הסטטוס not_found
        verdict = "PASS" if out["status"] == "not_found" else "CHECK"
    else:  # failure — documented on purpose; there is no "pass"
        # כשל מתועד — אין "מעבר", זו מגבלה מכוונת
        verdict = "DOCUMENTED FAILURE"
    # מחזירים את הפלט, התשובה והפסק
    return out, answer, verdict


# בלוק שמורץ בהרצה ישירה — מריץ את כל התרחישים וכותב קובץ ראיות
def main():
    # טוענים את הטבלה
    df = agent.load_table()
    # שורות הפתיחה של קובץ ה-Markdown
    lines = ["# Stage 14 — Test results & demo cases\n",
             f"_Classifier model: `{agent.MODEL}` · Report model: `{report.REPORT_MODEL}`_\n",
             "Each case runs the full pipeline: free text -> agent (intent+filters) "
             "-> backend function -> natural-language report.\n"]

    # קו מפריד בקונסולה
    print("=" * 72)
    # עוברים על כל תרחיש
    for sc in SCENARIOS:
        # מריצים אותו
        out, answer, verdict = _run_one(sc, df)
        # console
        # מדפיסים את הפסק לקונסולה
        print(f"[{sc['id']}] {verdict:18s} {sc['title']}")
        # מדפיסים את השאילתה
        print(f"     Q: {sc['query']}")
        # מדפיסים את הסטטוס והכוונה
        print(f"     -> status={out['status']} intent={out['intent']}")
        # markdown
        # מוסיפים כותרת תרחיש ל-Markdown
        lines.append(f"\n## {sc['id']} · {sc['title']}  — **{verdict}**\n")
        # השאילתה
        lines.append(f"- **Query:** {sc['query']}")
        # הכוונה שסווגה והסטטוס
        lines.append(f"- **Classified intent:** `{out['intent']}` (status: `{out['status']}`)")
        # אם תקין — מוסיפים את הכוונה הצפויה
        if sc["kind"] == "valid":
            lines.append(f"- **Expected intent:** {' / '.join(sc['expect_intent'])}")
        # תשובת הסוכן (בפורמט ציטוט)
        lines.append(f"- **Agent answer:**\n\n  > {answer.strip().replace(chr(10), chr(10)+'  > ')}\n")
        # אם זה כשל — מוסיפים את ההסבר מדוע (לפרק 7)
        if sc["kind"] == "failure":
            lines.append(f"- **Why it fails (for ch.7):** {sc['why']}\n")
    # קו מפריד
    print("=" * 72)

    # יוצרים את תיקיית הפלט אם אינה קיימת
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    # כותבים את קובץ הראיות
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    # מדפיסים היכן נכתב
    print(f"\nEvidence written to: {OUT_MD.relative_to(ROOT)}")


# נקודת כניסה
if __name__ == "__main__":
    main()
