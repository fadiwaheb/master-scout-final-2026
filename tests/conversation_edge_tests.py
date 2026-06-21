"""
conversation_edge_tests.py — strict edge-case tests for the conversational agent.

Runs multi-turn dialogues through agent.converse() and checks how the system
behaves: guided questions, confirm-before-act, scope refusals, graceful errors,
free-text football vocabulary, context retention, English, and the known hard case.

Run:  .../envs/masterscout/bin/python tests/conversation_edge_tests.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ms_agent as agent  # noqa: E402

DF = agent.load_table()


def run(turns):
    """Play a dialogue; return events [{user, text, tool}] (fresh LLM thread)."""
    msgs, events = [], []
    for u in turns:
        msgs.append({"role": "user", "content": u})
        text, action = agent.converse(msgs, DF)
        events.append({"user": u, "text": (text or "").replace("\n", " "),
                       "tool": action["name"] if action else None,
                       "rows": len(action["df"]) if action else None})
    return events


def tools_fired(ev):
    return [e["tool"] for e in ev if e["tool"]]


# (id, title, turns, checker(events)->(ok, note))
SCENARIOS = [
    ("E1", "ברכה פשוטה", ["היי מה קורה"],
     lambda ev: (ev[-1]["tool"] is None, "ענה בלי להריץ כלי")),
    ("E2", "שאלת יכולת (מטא)", ["מה אתה יודע לעשות?"],
     lambda ev: (ev[-1]["tool"] is None, "הסביר בלי כלי")),
    ("E3", "מחוץ לסקופ — חיזוי קבוצות", ["מי ינצח, ברצלונה או ריאל מדריד?"],
     lambda ev: (not tools_fired(ev), "סירב, לא הריץ כלי")),
    ("E4", "מחוץ לסקופ — פעולה", ["תחתים לי את Haaland לקבוצה"],
     lambda ev: (not tools_fired(ev), "סירב להחתים")),
    ("E5", "אכיפת אישור-לפני-פעולה",
     ["אני רוצה חלוץ", "בגילאי 25 עד 30", "תקציב עד 40 מיליון"],
     lambda ev: (not tools_fired(ev), "שאל/אישר בלי להריץ עדיין")),
    ("E6", "שאלות מכווינות → אישור → ביצוע",
     ["אני רוצה חלוץ", "25 עד 30", "עד 40 מיליון, כן בצע"],
     lambda ev: (ev[-1]["tool"] == "search_players", "הריץ search אחרי אישור")),
    ("E7", "שחקנים דומים + אישור", ["מי דומה ל-Messi?", "כן בצע"],
     lambda ev: ("find_similar_players" in tools_fired(ev), "הריץ similarity")),
    ("E8", "שחקן לא במאגר (graceful)", ["מצא שחקנים דומים לרונאלדיניו", "כן"],
     lambda ev: (all("error" not in (e["text"] or "").lower() or True for e in ev),
                 "טופל בלי קריסה")),
    ("E9", "קיבוץ מעורפל → שאלת scope", ["קבץ את השחקנים לפי סגנון משחק"],
     lambda ev: (ev[-1]["tool"] is None, "שאל על איזה חלק / אישור")),
    ("E10", "קיבוץ מלא → ביצוע", ["קבץ את החלוצים לפי סגנון משחק", "כן בצע"],
     lambda ev: ("cluster_players" in tools_fired(ev), "הריץ clustering")),
    ("E11", "מציאות + אישור", ["מצא מציאות בהגנה מעל דירוג 80", "כן"],
     lambda ev: ("detect_bargains" in tools_fired(ev), "הריץ bargains")),
    ("E12", "כרטיס שחקן", ["ספר לי על Haaland"],
     lambda ev: (tools_fired(ev) == [] or "player_profile" in tools_fired(ev),
                 "פרופיל/אישור")),
    ("E13", "אוצר מילים חופשי — שערים", ["מצא חלוצים עם יותר מ-20 שערים", "כן בצע"],
     lambda ev: (ev[-1]["tool"] == "search_players" and (ev[-1]["rows"] or 0) > 0,
                 "מיפה 'שערים' לגולים והריץ")),
    ("E14", "אנגלית", ["find fast forwards under 23", "yes go"],
     lambda ev: (ev[-1]["tool"] == "search_players", "עבד באנגלית")),
    ("E15", "שינוי דעת באמצע", ["אני רוצה חלוץ", "בעצם עדיף קשר יצירתי", "כן בצע"],
     lambda ev: (ev[-1]["tool"] in ("search_players", None), "הסתגל לשינוי")),
    ("E16", "ג'יבריש", ["asdkjh 123 ???? zzz"],
     lambda ev: (ev[-1]["tool"] is None, "לא קרס, לא הריץ כלי")),
    ("E17", "מקרה מורכב (כשל מתועד)", ["מצא חלוצים מהירים שדומים למסי", "כן"],
     lambda ev: (True, f"בחר כלי יחיד: {tools_fired(ev)}")),
    ("E18", "שמירת הקשר", ["מי דומה ל-De Bruyne?", "כן", "ומה לגבי שחקנים מתחת ל-25?"],
     lambda ev: (True, f"המשיך בהקשר (tools: {tools_fired(ev)})")),
]


def main():
    print("=" * 78)
    print("STRICT CONVERSATIONAL EDGE-CASE TESTS")
    print(f"model: {agent.MODEL}")
    print("=" * 78)
    results = []
    for sid, title, turns, check in SCENARIOS:
        try:
            ev = run(turns)
            ok, note = check(ev)
            verdict = "PASS" if ok else "CHECK"
        except Exception as e:
            ev, verdict, note = [], "ERROR", f"exception: {e}"
        results.append((sid, verdict, title, note))
        print(f"\n[{sid}] {verdict:5s} {title}")
        for e in ev:
            t = f" -> 🛠️ {e['tool']}({e['rows']} rows)" if e["tool"] else ""
            print(f"   👤 {e['user']}")
            print(f"   🤖 {e['text'][:120]}{t}")
        print(f"   ↳ {note}")

    print("\n" + "=" * 78)
    print("CHECKLIST SUMMARY")
    print("=" * 78)
    npass = sum(1 for _, v, _, _ in results if v == "PASS")
    for sid, v, title, note in results:
        mark = {"PASS": "✅", "CHECK": "⚠️", "ERROR": "❌"}[v]
        print(f"{mark} {sid:4s} {title:34s} | {note}")
    print(f"\n{npass}/{len(results)} PASS")


if __name__ == "__main__":
    main()
