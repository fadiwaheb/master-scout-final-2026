"""
app.py — Stage 15. The live Streamlit app (the public link the lecturer opens).

A natural-language chat agent (OpenAI function-calling):
  the LLM holds a free-flowing conversation, asks guiding questions, CONFIRMS
  before acting, and only then calls one of our Python tools (the ML algorithms).
  The whole chat history is sent every turn, so context is never lost.

Design: Academron dark glassmorphism — azure (#10d0f0) on near-black, Heebo font.
Hebrew-first RTL; player names and data tables stay LTR.
The OpenAI key is read from Streamlit Secrets (cloud) or .env (local).
"""

from pathlib import Path
import base64
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

for _k in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_REPORT_MODEL"):
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

import ms_agent as agent   # noqa: E402  (unique name — avoids clashing with any 'agent' package)
import workingset          # noqa: E402
import clustering          # noqa: E402
import external            # noqa: E402

# ---------------------------------------------------------------------------
# Page + theme (Academron dark glassmorphism)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Master Scout", layout="wide",
                   initial_sidebar_state="collapsed",
                   page_icon=str(ROOT / "assets" / "favicon.png"))

# custom-designed vector icons (magnifying-glass scout + person) on an azure plate
AVATARS = {"user": str(ROOT / "assets" / "avatar_user.png"),
           "assistant": str(ROOT / "assets" / "avatar_bot.png")}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
:root{
  --abyss:#03060d; --deep:#070b17; --elev1:#0b1224; --elev2:#0f1830;
  --azure:#10d0f0; --azure-2:#38d6f5; --azure-3:#7fe6fb; --azure-deep:#0070c0;
  --navy:#15233a; --txt:#f4f7fb; --txt2:#c5ccda; --txt3:#8a94ab;
  --glass:rgba(14,26,52,0.55); --glass-bd:rgba(56,214,245,0.18);
  --grad:linear-gradient(135deg,#10d0f0 0%,#00a0e0 45%,#0070c0 100%);
}
html, body, [class*="css"], .stApp, p, li, label, span, div{
  font-family:'Heebo', system-ui, sans-serif;
}
.stApp{
  background:
    radial-gradient(40rem 40rem at 80% 8%, rgba(16,208,240,0.10), transparent 60%),
    radial-gradient(46rem 46rem at 12% 92%, rgba(0,112,192,0.10), transparent 62%),
    var(--abyss);
}
.block-container{ padding-top:2rem; max-width:1100px; }
/* clean, focused layout — hide the top toolbar/header and the sidebar entirely */
[data-testid="stHeader"]{ display:none !important; }
[data-testid="stToolbar"]{ display:none !important; }
[data-testid="stSidebar"]{ display:none !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{ display:none !important; }
[data-testid="stDecoration"]{ display:none !important; }

/* custom chat avatars — premium azure glyphs instead of the default red/orange */
[data-testid="stChatMessageAvatar"]{
  background:radial-gradient(circle at 50% 38%, #102036 0%, #070b17 100%) !important;
  border:1.5px solid var(--azure); box-shadow:0 0 18px -2px rgba(16,208,240,.7);
  overflow:visible; position:relative; }
[data-testid="stChatMessageAvatar"] svg{ display:none; }     /* drop default icon */
[data-testid="stChatMessageAvatar"] img{ padding:5px; object-fit:contain;
  width:100%; height:100%; }
/* the user's name appears under their avatar */
.stChatMessage:has(.ms-user-tag) [data-testid="stChatMessageAvatar"]::after{
  content:var(--ms-username, ""); position:absolute; top:110%; left:50%;
  transform:translateX(-50%); font-size:.58rem; font-weight:700;
  color:var(--azure-3); white-space:nowrap; letter-spacing:.2px; }

/* name-gate (welcome screen) */
.ms-gate{ max-width:420px; margin:1.5rem auto 0; text-align:center; }
.ms-gate h3{ color:var(--txt); font-weight:700; margin-bottom:.2rem; }
.ms-gate p{ color:var(--txt2); font-size:.9rem; margin-bottom:.6rem; }
[data-testid="stTextInput"] input{ direction:rtl; text-align:center;
  font-size:1.05rem; font-weight:500; }

/* light decorative chat peek on the welcome screen */
.ms-preview{ max-width:520px; margin:1.7rem auto 0; opacity:.5; pointer-events:none; }
.ms-pv-row{ display:flex; margin:7px 0; }
.ms-pv{ display:inline-block; padding:.5rem .9rem; border-radius:13px; font-size:.85rem;
  direction:rtl; max-width:82%; }
.ms-pv.user{ margin-left:auto; color:var(--txt);
  background:linear-gradient(135deg,rgba(16,208,240,.17),rgba(56,214,245,.06));
  border:1px solid rgba(56,214,245,.4); }
.ms-pv.bot{ margin-right:auto; color:var(--txt2);
  background:linear-gradient(135deg,rgba(13,30,64,.8),rgba(7,14,30,.72));
  border:1px solid rgba(56,214,245,.22); }

/* entrance animation — spinning, growing ball */
.ms-enter{ position:fixed; inset:0; z-index:99999; display:flex; flex-direction:column;
  align-items:center; justify-content:center; background:var(--abyss); }
.ms-enter-ball{ font-size:4.4rem; filter:drop-shadow(0 0 34px rgba(16,208,240,.85));
  animation:enterSpin 1.4s cubic-bezier(.2,.8,.2,1) forwards; }
.ms-enter-text{ margin-top:1.2rem; color:var(--azure-3); font-size:1.5rem; font-weight:800;
  opacity:0; animation:enterFade .7s ease .55s forwards; }
@keyframes enterSpin{ 0%{ transform:rotate(0) scale(.25); opacity:0 }
  55%{ opacity:1 } 100%{ transform:rotate(720deg) scale(1.7); opacity:1 } }
@keyframes enterFade{ from{ opacity:0; transform:translateY(8px) } to{ opacity:1; transform:none } }
.ms-top{ display:flex; justify-content:center; margin:.1rem 0 0; }
.ms-logo{ height:300px; max-width:96%; display:block;
  filter:drop-shadow(0 8px 34px rgba(16,208,240,.45)); }
.ms-tagline{ text-align:center; color:var(--txt2); font-size:.95rem;
  margin:0 0 .35rem; font-weight:400; }
.ms-tagline b{ color:var(--azure-3); font-weight:600; }
.ms-credit{ text-align:center; color:var(--txt3); font-size:.78rem;
  letter-spacing:.3px; margin:0 0 1rem; }
.ms-credit b{ color:var(--azure-3); font-weight:600; }
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] ol{
  direction:rtl; text-align:right; padding-right:1.3em; padding-left:0; }
[data-testid="stMarkdownContainer"] li{ direction:rtl; text-align:right; }
[data-testid="stMarkdownContainer"]{ direction:rtl; text-align:right; }
[data-testid="stChatInput"] textarea{ direction:rtl; text-align:right; }
.stButton>button{ direction:rtl; }
.stExpander summary{ direction:rtl; text-align:right; }
code, pre, .stDataFrame, .stTable,
.stDataFrame [data-testid="stMarkdownContainer"]{ direction:ltr; text-align:left;
  font-family:'Fira Code', monospace; }
.stChatMessage{
  background:var(--glass) !important; border:1px solid var(--glass-bd);
  border-radius:14px; backdrop-filter:blur(8px);
}
/* assistant (the agent) bubble — deep blue */
.stChatMessage:has(span.ms-bot-tag){
  background:linear-gradient(135deg, rgba(13,30,64,.80), rgba(7,14,30,.72)) !important;
  border-color:rgba(56,214,245,.22) !important; }
/* user bubble — light turquoise (תכלת) */
.stChatMessage:has(span.ms-user-tag){
  background:linear-gradient(135deg, rgba(16,208,240,.17), rgba(56,214,245,.06)) !important;
  border-color:rgba(56,214,245,.50) !important;
  box-shadow:0 0 0 1px rgba(56,214,245,.12) inset; }
span.ms-bot-tag, span.ms-user-tag{ display:none; }
[data-testid="stSidebar"]{ background:var(--deep); border-inline-end:1px solid var(--glass-bd); }
[data-testid="stSidebar"] *{ direction:rtl; text-align:right; }
.stButton>button{
  background:rgba(16,208,240,.08); color:var(--azure-3);
  border:1px solid var(--glass-bd); border-radius:999px; font-weight:500;
  transition:all .18s ease; font-size:.85rem;
}
.stButton>button:hover{ background:rgba(16,208,240,.18); border-color:var(--azure); color:#fff; }
.stDataFrame thead tr th{ background:var(--elev2) !important; color:var(--azure-3) !important; }
.ms-note{ color:var(--txt3); font-size:.82rem; line-height:1.6; }
hr{ border-color:var(--glass-bd); }

/* ========== animations & polish ========== */
@keyframes fadeUp   { from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none} }
@keyframes glowPulse{ 0%,100%{filter:drop-shadow(0 8px 26px rgba(16,208,240,.35))}
                      50%{filter:drop-shadow(0 12px 44px rgba(16,208,240,.75))} }
@keyframes bgDrift  { 0%{background-position:0% 0%,100% 100%,0 0} 100%{background-position:8% 5%,92% 95%,0 0} }
@keyframes popIn    { 0%{opacity:0; transform:scale(.5)} 70%{transform:scale(1.12)} 100%{opacity:1; transform:scale(1)} }
@keyframes ringspin { to{ transform:rotate(360deg) } }

/* chat bubbles glide in + lift on hover */
.stChatMessage{ animation:fadeUp .45s cubic-bezier(.16,1,.3,1) both;
  transition:transform .2s ease, box-shadow .2s ease; }
.stChatMessage:hover{ transform:translateY(-2px);
  box-shadow:0 16px 44px -20px rgba(16,208,240,.5); }

/* living background + pulsing logo */
.stApp{ background-size:140% 140%, 150% 150%, auto;
  animation:bgDrift 20s ease-in-out infinite alternate; }
.ms-logo{ animation:glowPulse 3.4s ease-in-out infinite; }

/* azure custom scrollbar */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:linear-gradient(var(--azure-deep),var(--azure));
  border-radius:10px; border:2px solid var(--abyss); }
::-webkit-scrollbar-thumb:hover{ background:var(--azure-2); }

/* dataframe + buttons micro-interactions */
.stDataFrame{ transition:transform .2s ease, box-shadow .2s ease; border-radius:10px; }
.stDataFrame:hover{ transform:translateY(-2px);
  box-shadow:0 16px 44px -22px rgba(16,208,240,.45); }
.stButton>button:active{ transform:scale(.96); }

/* ---- upgraded player card ---- */
.ms-card-head{ display:flex; align-items:center; gap:16px; margin:.2rem 0 .7rem;
  direction:rtl; }
.ms-photo-wrap{ position:relative; width:92px; height:92px; flex:none; }
.ms-photo-wrap::before{ content:""; position:absolute; inset:-5px; border-radius:50%;
  background:conic-gradient(from 0deg,var(--azure),var(--azure-deep),var(--azure-3),var(--azure));
  animation:ringspin 6s linear infinite; filter:blur(1px); opacity:.9; }
.ms-photo{ position:relative; width:92px; height:92px; border-radius:50%; object-fit:cover;
  background:var(--elev2); border:3px solid var(--deep);
  box-shadow:0 10px 30px -8px rgba(16,208,240,.6); animation:popIn .55s cubic-bezier(.16,1,.3,1) both; }
.ms-card-meta{ flex:1; text-align:right; }
.ms-card-name{ font-size:1.3rem; font-weight:800; color:var(--txt); line-height:1.2; }
.ms-card-club{ color:var(--txt2); font-size:.92rem; margin-top:3px; }
.ms-flag,.ms-badge-img{ height:20px; vertical-align:middle; border-radius:3px;
  margin-inline-start:7px; }
.ms-ovr{ width:58px; height:58px; flex:none; border-radius:50%; display:flex;
  align-items:center; justify-content:center; font-weight:800; font-size:1.45rem;
  color:#03060d; background:var(--ovr-c,#10d0f0); box-shadow:0 0 22px -2px var(--ovr-c,#10d0f0);
  animation:popIn .65s cubic-bezier(.16,1,.3,1) both; }
/* ---- FIFA-style (FUT) player card ---- */
.fut-card{ width:244px; padding:24px 18px 16px; text-align:center;
  position:relative; margin:0 auto; direction:ltr;
  /* chamfered octagon — a card/badge silhouette, not a plain rectangle */
  clip-path:polygon(5% 0,95% 0,100% 5%,100% 88%,82% 100%,18% 100%,0 88%,0 5%);
  /* drop-shadow (not box-shadow) so the glow follows the clipped shape */
  filter:drop-shadow(0 16px 26px rgba(0,0,0,.6)) drop-shadow(0 0 12px rgba(16,208,240,.3));
  animation:popIn .55s cubic-bezier(.16,1,.3,1) both; }
.fut-card::after{ content:""; position:absolute; inset:6px;
  clip-path:polygon(5% 0,95% 0,100% 5%,100% 88%,82% 100%,18% 100%,0 88%,0 5%);
  border:1.5px solid rgba(0,0,0,.18); pointer-events:none; }
.fut-gold  { background:linear-gradient(165deg,#fbe899 0%,#e6c24c 42%,#bb8a13 100%); color:#3a2c04; }
.fut-silver{ background:linear-gradient(165deg,#f3f6f9 0%,#cfd6dd 45%,#9ba4ad 100%); color:#2a2f36; }
.fut-bronze{ background:linear-gradient(165deg,#edbd8b 0%,#cf8338 45%,#9c5a22 100%); color:#3a230d; }
.fut-top{ position:absolute; top:22px; left:20px; line-height:1; text-align:left; z-index:2; }
.fut-ovr{ font-size:2.15rem; font-weight:900; }
.fut-pos{ font-size:.82rem; font-weight:800; opacity:.85; margin-top:1px; }
.fut-photo{ height:138px; margin-top:4px; display:flex; align-items:flex-end; justify-content:center; }
.fut-photo img{ height:138px; object-fit:contain; object-position:bottom;
  filter:drop-shadow(0 6px 8px rgba(0,0,0,.35)); }
.fut-noimg{ height:138px; display:flex; align-items:center; justify-content:center;
  font-size:3rem; opacity:.5; }
.fut-name{ font-weight:900; font-size:1.2rem; letter-spacing:.6px; text-transform:uppercase;
  border-top:2px solid rgba(0,0,0,.28); padding-top:5px; margin-top:3px; }
.fut-stats{ display:grid; grid-template-columns:repeat(6,1fr); gap:1px; margin-top:7px; }
.fut-stats > div{ display:flex; flex-direction:column; line-height:1.05; }
.fut-stats b{ font-size:1.02rem; font-weight:900; }
.fut-stats span{ font-size:.6rem; font-weight:800; opacity:.78; letter-spacing:.3px; }
.fut-bottom{ margin-top:8px; border-top:2px solid rgba(0,0,0,.22); padding-top:6px; }
.fut-bottom img{ height:20px; border-radius:2px; box-shadow:0 1px 3px rgba(0,0,0,.4); }
.fut-facts{ text-align:center; color:var(--txt2); font-size:.82rem; margin-top:.5rem; }

@media (prefers-reduced-motion: reduce){ *{ animation:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

_LOGO_B64 = base64.b64encode((ROOT / "assets" / "logo.png").read_bytes()).decode()
st.markdown(
    f"""
    <div class="ms-top">
      <img class="ms-logo" src="data:image/png;base64,{_LOGO_B64}" alt="Master Scout"/>
    </div>
    <p class="ms-tagline" dir="rtl">משלב פרופיל סטטי (<b>EA FC24</b>) עם ביצועים
    אמיתיים מהמגרש (<b>Football Events</b>) · שיחה חופשית, המלצה מנומקת · <b>PoC</b></p>
    <p class="ms-credit">© 2026 <b>Fadi Waheb</b> · All rights reserved · כל הזכויות שמורות</p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Name gate — ask for a name once, then greet the user personally
# ---------------------------------------------------------------------------
if "username" not in st.session_state:
    # entrance animation: a spinning, growing ball, then into the chat
    if st.session_state.get("entering"):
        _name = st.session_state.entering
        st.markdown(f"<div class='ms-enter'><div class='ms-enter-ball'>⚽</div>"
                    f"<div class='ms-enter-text'>ברוכים הבאים, {_name}! 🚀</div></div>",
                    unsafe_allow_html=True)
        import time as _t
        _t.sleep(1.4)
        st.session_state.username = _name
        del st.session_state["entering"]
        st.rerun()

    st.markdown("<div class='ms-gate'><h3>ברוכים הבאים 👋</h3>"
                "<p>איך לקרוא לך? כך אוכל לפנות אליך באופן אישי.</p></div>",
                unsafe_allow_html=True)
    _c = st.columns([1, 2, 1])[1]
    with _c:
        with st.form("name_gate", border=False):
            _nm = st.text_input("שם", label_visibility="collapsed", placeholder="הקלידו שם…")
            _go = st.form_submit_button("כניסה ⚽", use_container_width=True)
        if _go and _nm.strip():
            st.session_state.entering = _nm.strip()[:24]
            st.rerun()
    # a light, decorative peek at the conversation
    st.markdown(
        "<div class='ms-preview'>"
        "<div class='ms-pv-row'><span class='ms-pv user'>מצא חלוצים מהירים מתחת לגיל 23 💨</span></div>"
        "<div class='ms-pv-row'><span class='ms-pv bot'>הנה 3 חלוצים מהירים ומבטיחים בתקציב שלך ⚽</span></div>"
        "<div class='ms-pv-row'><span class='ms-pv user'>מי דומה ל-Messi? 🐐</span></div>"
        "<div class='ms-pv-row'><span class='ms-pv bot'>הדומים ביותר בסגנון: Dybala, Martial, Ben Yedder 📊</span></div>"
        "</div>", unsafe_allow_html=True)
    st.stop()

# make the name available to the CSS (shown under the user avatar)
_uname = re.sub(r'["<>]', "", st.session_state.username)
st.markdown(f"<style>:root{{--ms-username:\"{_uname}\"}}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + key
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_table():
    return agent.load_table()


def key_ready() -> bool:
    try:
        agent._get_client()
        return True
    except Exception:
        return False


df = get_table()


def get_full_row(name):
    """Full-column row for a player by short_name (for the card / radar)."""
    m = df[df["short_name"].astype(str) == str(name)]
    return m.sort_values("overall", ascending=False).iloc[0] if len(m) else None


# ---------------------------------------------------------------------------
# Charts + player card
# ---------------------------------------------------------------------------
_PALETTE = ["#10d0f0", "#f5c26b", "#9fe7c0", "#e0729f", "#7fe6fb", "#b39ddb"]


def make_cluster_scatter(xy, cluster_ids, label_map):
    """2D PCA scatter of the clusters (azure palette, one colour per style)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for i, cid in enumerate(sorted(set(cluster_ids))):
        m = cluster_ids == cid
        ax.scatter(xy[m, 0], xy[m, 1], s=15, alpha=.7, edgecolors="none",
                   color=_PALETTE[i % len(_PALETTE)], label=label_map.get(cid, str(cid)))
    ax.set_xlabel("Principal Component 1", color="#c5ccda", fontsize=9)
    ax.set_ylabel("Principal Component 2", color="#c5ccda", fontsize=9)
    ax.tick_params(colors="#8a94ab", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor="#c5ccda")
    fig.tight_layout()
    return fig


_RADAR_ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
_RADAR_LABELS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]


def make_radar(players):
    """Radar/spider chart of the 6 core attributes for one or more players."""
    angles = np.linspace(0, 2 * np.pi, len(_RADAR_ATTRS), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.4, 4.4), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for i, p in enumerate(players):
        vals = [float(p[a]) for a in _RADAR_ATTRS]
        vals += vals[:1]
        c = _PALETTE[i % len(_PALETTE)]
        ax.plot(angles, vals, color=c, linewidth=2, label=str(p["short_name"]))
        ax.fill(angles, vals, color=c, alpha=.22)
        # value number at each vertex (single-player card only — avoids clutter)
        if len(players) == 1:
            for ang, v in zip(angles[:-1], vals[:-1]):
                ax.annotate(str(int(round(v))), xy=(ang, min(v + 7, 99)),
                            ha="center", va="center", color="#7fe6fb",
                            fontsize=10, fontweight="bold")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(_RADAR_LABELS, color="#c5ccda", fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#5e6b85", fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(color="#26324a")
    ax.spines["polar"].set_color("#26324a")
    if len(players) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8,
                  frameon=False, labelcolor="#c5ccda")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def enrich_media(name):
    """Photo + national flag for a player, fetched live from EA's ratings API
    (so even players that ARE in our data get a portrait + flag). Cached."""
    try:
        return external.ea_media(name)
    except Exception:
        return {}


def _val(p, key):
    v = p.get(key) if hasattr(p, "get") else None
    return v if (v is not None and pd.notna(v)) else None


def _tier(v):
    if v is None or v >= 80:
        return "gold" if v is not None else "silver"
    return "silver" if v >= 70 else "bronze"


_POS_ABBR = {"Forward": "FWD", "Midfielder": "MID", "Defender": "DEF", "GK": "GK"}
_FUT_STATS = [("PAC", "pace"), ("SHO", "shooting"), ("PAS", "passing"),
              ("DRI", "dribbling"), ("DEF", "defending"), ("PHY", "physic")]


def render_player_card(p):
    """A FIFA-style (FUT) card — gold (80+) / silver (70-79) / bronze (<70) — with
    the player photo, OVR, position, 6 stats and the national flag; next to a
    compact radar. Photo/flag enriched live from EA for players in our data."""
    name = str(p["short_name"])
    av = _val(p, "avatar_url")
    flag = _val(p, "nationality_flag_url")
    if av is None or flag is None:
        m = enrich_media(name)
        av = av or m.get("avatar_url")
        flag = flag or m.get("flag_url")
    ovr = int(p["overall"]) if pd.notna(p.get("overall")) else None
    tier = _tier(ovr)
    pos = _POS_ABBR.get(p.get("position_group"), p.get("position_group") or "")
    last = re.sub(r"^[A-Z]\.\s*", "", name).strip() or name

    def s(k):
        v = p.get(k)
        return int(v) if (v is not None and pd.notna(v)) else "—"
    stats_html = "".join(f"<div><b>{s(k)}</b><span>{lbl}</span></div>" for lbl, k in _FUT_STATS)
    photo_html = f"<img src='{av}'>" if av else "<div class='fut-noimg'>⚽</div>"
    flag_html = f"<div class='fut-bottom'><img src='{flag}'></div>" if flag else ""
    card = (f"<div class='fut-card fut-{tier}'>"
            f"<div class='fut-top'><div class='fut-ovr'>{ovr if ovr is not None else '—'}</div>"
            f"<div class='fut-pos'>{pos}</div></div>"
            f"<div class='fut-photo'>{photo_html}</div>"
            f"<div class='fut-name'>{last}</div>"
            f"<div class='fut-stats'>{stats_html}</div>{flag_html}</div>")

    # textual details folded in beneath
    facts = []
    if pd.notna(p.get("age")):
        facts.append(f"גיל {int(p['age'])}")
    if pd.notna(p.get("potential")):
        facts.append(f"פוטנציאל {int(p['potential'])}")
    if pd.notna(p.get("value_eur")):
        facts.append(f"שווי €{int(p['value_eur']):,}")
    if p.get("preferred_foot"):
        facts.append(f"רגל {p['preferred_foot']}")
    if p.get("nationality_name"):
        facts.append(str(p["nationality_name"]))
    if p.get("has_event_data"):
        facts.append(f"⚽ {int(p.get('total_goals', 0))} גולים")
        facts.append(f"דאבלים {int(p.get('matches_with_2_plus_goals', 0))}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(make_radar([p]), use_container_width=True)
    with c2:
        st.markdown(card, unsafe_allow_html=True)
    st.markdown(f"<div class='fut-facts'>{' · '.join(facts)}</div>", unsafe_allow_html=True)


# compact "about / sources / ethics" — collapsed, so the top stays clean
with st.expander("ℹ️ אודות · מקורות נתונים · אתיקה"):
    st.markdown(
        f"<span class='ms-note'>נתונים: {len(df):,} שחקני FC24 · "
        f"{int(df['has_event_data'].sum()):,} עם נתוני אירועי משחק (2012–2017).<br>"
        f"<b>סדר מקורות:</b> מאגר ראשי (FC24) → משני (אירועים) → "
        f"EA Sports FC הרשמי (API חי) → ידע המודל.<br>"
        f"גבול: שחקנים בלבד — לא קבוצות, לא חיזוי, לא הימורים. הסוכן ממליץ בלבד "
        f"ועוצר לפני פעולה. PoC.<br>"
        f"⚖️ <b>אתיקה:</b> המערכת משתמשת במודלים של OpenAI; שאילתות נשלחות לשירות "
        f"חיצוני ועשויות לשמש לשיפור המודלים. נתוני FC24 סובייקטיביים וחלקיים.</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Conversation state
#   history : display items {role, content, artifact?}
#   llm     : OpenAI-format messages (incl. tool calls) sent to the model
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    _greet = agent.greeting_message("he").replace("היי!", f"היי {_uname}!", 1)
    st.session_state.history = [{"role": "assistant", "content": _greet}]
    st.session_state.llm = [{"role": "system",
        "content": f"The user's name is {_uname}. Address them warmly by name now "
                   f"and then (especially when greeting), in their language."}]

if not key_ready():
    st.warning("מפתח OpenAI לא נמצא. מקומית: הוסיפו ל-`.env`. "
               "ב-Streamlit Cloud: Settings → Secrets → `OPENAI_API_KEY`.")

_PLAYER_TOOLS = ("search_players", "find_similar_players", "detect_bargains")


_SRC_LABEL = {"primary": "מאגר ראשי (FC24) + משני (אירועים)",
              "ea": "EA Sports FC — דירוגים רשמיים (API חי)",
              "web": "ידע המודל (חיפוש בדומיין)"}


def _src_footer(src, n=None):
    lbl = _SRC_LABEL.get(src, "")
    pre = f"📋 {n} שורות · " if n is not None else ""
    if lbl:
        st.markdown(f"<span class='ms-note'>{pre}📊 מבוסס על: {lbl}</span>",
                    unsafe_allow_html=True)
    elif pre:
        st.markdown(f"<span class='ms-note'>{pre.rstrip(' ··')}</span>",
                    unsafe_allow_html=True)


def render_artifact(art):
    name, dfa, src = art["name"], art["df"], art.get("source")
    # one player card (not a disambiguation list)
    if name == "player_profile" and not art.get("disambig"):
        if src == "ea":
            st.info("ℹ️ שחקן זה אינו במאגרים שלנו — הנתונים והתמונה נשלפו **בזמן אמת** "
                    "מהדירוגים הרשמיים של EA Sports FC (ea.com/ratings).")
        elif src == "web":
            st.info("⚠️ שחקן זה אינו במאגרים שלנו ולא נמצא ב-EA — הכרטיס נבנה בעזרת "
                    "המודל לפי ידע על השחקן (חיפוש בדומיין).")
        render_player_card(dfa.iloc[0])
        _src_footer(src)
        return
    if name == "cluster_players":
        st.dataframe(dfa, use_container_width=True, hide_index=True)
        sc = art.get("scatter")
        if sc:
            st.pyplot(make_cluster_scatter(np.array(sc["xy"]), np.array(sc["cids"]),
                                           sc["label_map"]), use_container_width=True)
        _src_footer(src)
        return
    # a table: players list OR disambiguation candidates
    st.dataframe(dfa, use_container_width=True, hide_index=True)
    if art.get("disambig"):
        st.markdown("<span class='ms-note'>נמצאו כמה שחקנים בשם הזה — כתבו את "
                    "השם המלא של זה שמעניין אתכם.</span>", unsafe_allow_html=True)
    _src_footer(src, len(dfa))


# index of the most recent players table -> the inline card picker sits there
_last_pidx = None
for _i, _m in enumerate(st.session_state.history):
    _a = _m.get("artifact")
    if (_a and _a["name"] in _PLAYER_TOOLS and not _a.get("disambig")
            and "short_name" in _a["df"].columns):
        _last_pidx = _i

# render the conversation so far (with the card picker inline at the latest result)
for _i, m in enumerate(st.session_state.history):
    with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
        _tag = "ms-bot-tag" if m["role"] == "assistant" else "ms-user-tag"
        st.markdown(f"<span class='{_tag}'></span>", unsafe_allow_html=True)
        if m.get("content"):
            st.markdown(m["content"])
        if m.get("artifact"):
            render_artifact(m["artifact"])
        if _i == _last_pidx:
            _names = ["—"] + m["artifact"]["df"]["short_name"].astype(str).tolist()
            _pick = st.selectbox("🔎 כרטיס שחקן — בחרו שחקן מהרשימה", _names, key=f"pick_{_i}")
            if _pick != "—":
                _row = get_full_row(_pick)
                if _row is not None:
                    render_player_card(_row)


# ---------------------------------------------------------------------------
# Chat input -> one conversational turn
# ---------------------------------------------------------------------------
if prompt := st.chat_input("כתבו לסוכן בשפה חופשית…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.llm.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown("<span class='ms-user-tag'></span>", unsafe_allow_html=True)
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown("<span class='ms-bot-tag'></span>", unsafe_allow_html=True)
        with st.spinner("חושב…"):
            try:
                text, action = agent.converse(st.session_state.llm, df)
            except Exception as e:
                text, action = f"מצטער, נתקלתי בשגיאה: {e}", None

    art = None
    if action:
        ex = action.get("extra", {}) or {}
        src = ex.get("source")
        if action["name"] == "cluster_players":
            try:
                xy, cids = clustering.cluster_xy(ex["labeled"])
                lm = {d["cluster_id"]: d["label"] for d in ex["descriptions"]}
                art = {"name": "cluster_players", "df": action["df"], "source": src,
                       "scatter": {"xy": xy.tolist(), "cids": cids.tolist(),
                                   "label_map": lm}}
            except Exception:
                art = {"name": "cluster_players", "df": action["df"], "source": src}
        else:
            art = {"name": action["name"], "df": action["df"], "source": src,
                   "disambig": ex.get("disambiguation", False)}

    st.session_state.history.append(
        {"role": "assistant", "content": text, "artifact": art})
    st.rerun()
