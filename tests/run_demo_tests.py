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

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ms_agent as agent      # noqa: E402
import report     # noqa: E402

OUT_MD = ROOT / "reports" / "14_test_results.md"


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


def _run_one(sc, df):
    out = agent.run_agent(sc["query"], df)
    # build the user-facing answer (report for ok, else the status message)
    answer = report.generate_report(out) if out["status"] == "ok" else out["message"]

    # decide PASS/observed for the summary
    if sc["kind"] == "valid":
        ok = out["status"] == "ok" and out["intent"] in sc["expect_intent"]
        verdict = "PASS" if ok else "CHECK"
    elif sc["kind"] == "refusal":
        verdict = "PASS" if out["status"] == "out_of_scope" else "CHECK"
    elif sc["kind"] == "clarify":
        verdict = "PASS" if out["status"] == "clarify" else "CHECK"
    elif sc["kind"] == "not_found":
        verdict = "PASS" if out["status"] == "not_found" else "CHECK"
    else:  # failure — documented on purpose; there is no "pass"
        verdict = "DOCUMENTED FAILURE"
    return out, answer, verdict


def main():
    df = agent.load_table()
    lines = ["# Stage 14 — Test results & demo cases\n",
             f"_Classifier model: `{agent.MODEL}` · Report model: `{report.REPORT_MODEL}`_\n",
             "Each case runs the full pipeline: free text -> agent (intent+filters) "
             "-> backend function -> natural-language report.\n"]

    print("=" * 72)
    for sc in SCENARIOS:
        out, answer, verdict = _run_one(sc, df)
        # console
        print(f"[{sc['id']}] {verdict:18s} {sc['title']}")
        print(f"     Q: {sc['query']}")
        print(f"     -> status={out['status']} intent={out['intent']}")
        # markdown
        lines.append(f"\n## {sc['id']} · {sc['title']}  — **{verdict}**\n")
        lines.append(f"- **Query:** {sc['query']}")
        lines.append(f"- **Classified intent:** `{out['intent']}` (status: `{out['status']}`)")
        if sc["kind"] == "valid":
            lines.append(f"- **Expected intent:** {' / '.join(sc['expect_intent'])}")
        lines.append(f"- **Agent answer:**\n\n  > {answer.strip().replace(chr(10), chr(10)+'  > ')}\n")
        if sc["kind"] == "failure":
            lines.append(f"- **Why it fails (for ch.7):** {sc['why']}\n")
    print("=" * 72)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEvidence written to: {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
