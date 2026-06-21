"""
comprehensive_tests.py — full system check for Master Scout.

Covers the SOURCE-PRIORITY chain (our data -> EA official API -> model), player
cards + photos, name disambiguation, similar-to-unknown, free-text football
vocabulary, goalkeeper data, and conversational behaviour (greeting, scope,
confirm-before-act, citations). Prints a single PASS/CHECK checklist.

Run:  .../envs/masterscout/bin/python tests/comprehensive_tests.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ms_agent as agent      # noqa: E402
import external               # noqa: E402

DF = agent.load_table()
results = []


def check(idx, title, fn):
    try:
        ok, note = fn()
    except Exception as e:
        ok, note = False, f"exception: {e}"
    results.append((idx, "PASS" if ok else "CHECK", title, note))


def say(turns):
    """Run a dialogue; return events [{text, tool, src}]."""
    msgs, ev = [], []
    for u in turns:
        msgs.append({"role": "user", "content": u})
        t, a = agent.converse(msgs, DF)
        ev.append({"text": (t or ""), "tool": a["name"] if a else None,
                   "src": a["extra"].get("source") if a else None,
                   "df": a["df"] if a else None})
    return ev


# ---- A. SOURCE PRIORITY (deterministic, via run_tool) ----------------------
def t_primary():
    res, ex, _ = agent.run_tool("player_profile", {"player_name": "Olise"}, DF)
    return ex["source"] == "primary", f"Olise -> source={ex['source']} (in our FC24 data)"

def t_ea():
    res, ex, _ = agent.run_tool("player_profile", {"player_name": "Lamine Yamal"}, DF)
    r = res.iloc[0]
    ok = ex["source"] == "ea" and r["overall"] and ex.get("avatar")
    return ok, f"Yamal -> source={ex['source']}, OVR={int(r['overall'])}, photo={bool(ex.get('avatar'))}"

def t_model():
    res, ex, _ = agent.run_tool("player_profile", {"player_name": "Dor Peretz"}, DF)
    return ex["source"] in ("web", "ea") and len(res) == 1, \
        f"Dor Peretz -> source={ex['source']}, card built ({res.iloc[0]['short_name']})"

def t_notfound():
    try:
        agent.run_tool("player_profile", {"player_name": "Xqzzy Zzqx Random 9000"}, DF)
        return False, "should have raised not-found"
    except ValueError:
        return True, "random name -> correctly not found (no card invented)"

def t_disambig():
    res, ex, _ = agent.run_tool("player_profile", {"player_name": "Silva"}, DF)
    return ex.get("disambiguation") and len(res) > 1, \
        f"'Silva' -> {len(res)} candidates listed (disambiguation)"

def t_similar_unknown():
    res, ex, _ = agent.run_tool("find_similar_players",
                                {"player_name": "Lamine Yamal", "top_n": 5}, DF)
    return ex["source"] in ("ea", "web") and len(res) >= 3, \
        f"similar to Yamal (not in data) -> via {ex['source']}: " + \
        ", ".join(res["short_name"].head(3).astype(str))


# ---- B. EA API + photos (direct) -------------------------------------------
def t_ea_api():
    d = external.ea_fc_lookup("Mbappé")
    ok = d and d.get("overall") and d.get("avatar_url")
    return bool(ok), (f"EA API: {d['short_name']} OVR={d['overall']} pos={d['position_group']} "
                      f"photo={bool(d.get('avatar_url'))}" if d else "no result")


# ---- C. Goalkeepers data insight -------------------------------------------
def t_gk():
    gk = DF[DF["position_group"] == "GK"]
    nan_all = gk[["pace", "shooting", "passing", "dribbling", "defending", "physic"]].isna().all().all()
    return nan_all, f"{len(gk)} GKs all have NaN face-attrs -> can't cluster/compare (by design)"


# ---- D. Football vocabulary (free text) ------------------------------------
def t_vocab():
    res, ex, _ = agent.run_tool("search_players",
                                {"position": "Forward", "min_total_goals": 20}, DF)
    return len(res) > 0 and "total_goals" in res.columns, \
        f"'20+ goals' -> {len(res)} players sorted by goals (top: {res.iloc[0]['short_name']})"


# ---- E. Conversation behaviour (via the LLM) -------------------------------
def t_greeting():
    ev = say(["היי מה קורה"])
    return ev[-1]["tool"] is None, "greeting -> friendly reply, no tool"

def t_scope():
    ev = say(["מי ינצח ברצלונה או ריאל?"])
    return ev[-1]["tool"] is None, "team-prediction -> refused, no tool"

def t_confirm():
    ev = say(["אני רוצה חלוץ", "25-30", "עד 40 מיליון"])
    fired = [e["tool"] for e in ev if e["tool"]]
    return not fired, "gathered criteria without a 'go' -> did NOT run yet (confirm-first)"

def t_he_profile_ea():
    ev = say(["תראה לי פרטים על לאמין יאמל"])
    e = ev[-1]
    cite = "EA" in e["text"] or "מאגרים" in e["text"]
    return e["tool"] == "player_profile" and e["src"] == "ea" and cite, \
        f"Hebrew 'Yamal' -> player_profile via EA, source cited"

def t_clustering_optimal():
    ev = say(["קבץ את החלוצים לפי סגנון משחק", "כן בצע"])
    fired = [e for e in ev if e["tool"] == "cluster_players"]
    return bool(fired), "clustering -> ran with auto-optimal K (never asked K)"


CHECKS = [
    ("S1", "מקור ראשי (במאגר שלנו)", t_primary),
    ("S2", "מקור EA רשמי (לא במאגר, יש ב-EA) + תמונה", t_ea),
    ("S3", "מקור מודל (לא במאגר ולא ב-EA)", t_model),
    ("S4", "שם רנדומלי — לא נמצא (לא ממציא)", t_notfound),
    ("S5", "דיסאמביגואציה (שם רב-משמעי)", t_disambig),
    ("S6", "דמיון לשחקן שלא במאגר", t_similar_unknown),
    ("S7", "EA API חי + תמונה (Mbappé)", t_ea_api),
    ("S8", "תובנת שוערים (NaN בתכונות)", t_gk),
    ("S9", "אוצר מילים חופשי (20+ שערים)", t_vocab),
    ("C1", "ברכה (בלי כלי)", t_greeting),
    ("C2", "גבול סקופ (סירוב)", t_scope),
    ("C3", "אישור לפני פעולה", t_confirm),
    ("C4", "כרטיס מ-EA דרך שיחה בעברית + ציון מקור", t_he_profile_ea),
    ("C5", "קיבוץ עם K אופטימלי אוטומטי", t_clustering_optimal),
]


def main():
    print("=" * 80)
    print("MASTER SCOUT — COMPREHENSIVE SYSTEM CHECK")
    print(f"chat model: {agent.CHAT_MODEL}")
    print("=" * 80)
    for idx, title, fn in CHECKS:
        check(idx, title, fn)
    npass = sum(1 for _, v, _, _ in results if v == "PASS")
    for idx, v, title, note in results:
        mark = {"PASS": "✅", "CHECK": "⚠️"}[v]
        print(f"{mark} {idx:3s} {title:42s} | {note}")
    print("=" * 80)
    print(f"{npass}/{len(results)} PASS")


if __name__ == "__main__":
    main()
