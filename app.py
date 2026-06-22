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

# ייבוא Path לעבודה עם נתיבי קבצים
from pathlib import Path
# ייבוא base64 להטמעת תמונות (לוגו/אווטרים) ישירות ב-HTML
import base64
# ייבוא os לקריאת/כתיבת משתני סביבה (מפתחות מ-secrets)
import os
# ייבוא re לביטויים רגולריים (ניקוי שם משתמש וכו')
import re
# ייבוא sys כדי להוסיף את תיקיית src לנתיב הייבוא
import sys

# ייבוא numpy לחישובים מספריים
import numpy as np
# ייבוא pandas לעבודה עם הטבלה
import pandas as pd
# ייבוא matplotlib לציור גרפים
import matplotlib
# מגדירים מנוע ציור ללא תצוגה (מתאים לשרת Streamlit)
matplotlib.use("Agg")
# ייבוא ממשק ה-pyplot לציור
import matplotlib.pyplot as plt
# ייבוא Streamlit — מסגרת אפליקציית הצ'אט
import streamlit as st

# שורש הפרויקט (תיקיית הקובץ)
ROOT = Path(__file__).resolve().parent
# מוסיפים את תיקיית src לנתיב הייבוא
sys.path.insert(0, str(ROOT / "src"))

# מעבירים מפתחות/שמות מודלים מ-Streamlit secrets אל משתני הסביבה
for _k in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_REPORT_MODEL"):
    # מנסים לקרוא כל מפתח מ-secrets (לא תמיד קיים)
    try:
        # אם המפתח קיים ב-secrets — מציבים אותו בסביבה
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        # אם אין secrets (הרצה מקומית) — מתעלמים
        pass

# מייבאים את שכבת הסוכן בשם ייחודי (כדי לא להתנגש בחבילת 'agent')
import ms_agent as agent   # noqa: E402  (unique name — avoids clashing with any 'agent' package)
# מייבאים את מנוע קבוצת העבודה
import workingset          # noqa: E402
# מייבאים את מודול הקיבוץ
import clustering          # noqa: E402
# מייבאים את מודול המקור החיצוני
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
/* משתני צבע גלובליים (פלטת ה-azure על רקע כמעט-שחור) של עיצוב Academron */
:root{
  --abyss:#03060d; --deep:#070b17; --elev1:#0b1224; --elev2:#0f1830;
  --azure:#10d0f0; --azure-2:#38d6f5; --azure-3:#7fe6fb; --azure-deep:#0070c0;
  --navy:#15233a; --txt:#f4f7fb; --txt2:#c5ccda; --txt3:#8a94ab;
  --glass:rgba(14,26,52,0.55); --glass-bd:rgba(56,214,245,0.18);
  --grad:linear-gradient(135deg,#10d0f0 0%,#00a0e0 45%,#0070c0 100%);
}
/* גופן בסיסי (Heebo) לכל אלמנטי הטקסט באפליקציה */
html, body, [class*="css"], .stApp, p, li, label, span, div{
  font-family:'Heebo', system-ui, sans-serif;
}
/* רקע האפליקציה — שני זוהרי azure רכים על רקע התהום הכהה */
.stApp{
  background:
    radial-gradient(40rem 40rem at 80% 8%, rgba(16,208,240,0.10), transparent 60%),
    radial-gradient(46rem 46rem at 12% 92%, rgba(0,112,192,0.10), transparent 62%),
    var(--abyss);
}
/* מיכל התוכן המרכזי — ריווח עליון ורוחב מרבי */
.block-container{ padding-top:2rem; max-width:1100px; }
/* clean, focused layout — hide the top toolbar/header and the sidebar entirely */
/* פריסה נקייה — מסתירים את סרגל הכלים, הכותרת והסרגל הצדדי של Streamlit */
[data-testid="stHeader"]{ display:none !important; }
[data-testid="stToolbar"]{ display:none !important; }
[data-testid="stSidebar"]{ display:none !important; }
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{ display:none !important; }
[data-testid="stDecoration"]{ display:none !important; }

/* אווטרים מותאמים לצ'אט — סמלי azure במקום ברירת המחדל האדומה/כתומה */
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

/* מסך הכניסה (שער השם) — בקשת שם המשתמש */
.ms-gate{ max-width:420px; margin:1.5rem auto 0; text-align:center; }
.ms-gate h3{ color:var(--txt); font-weight:700; margin-bottom:.2rem; }
.ms-gate p{ color:var(--txt2); font-size:.9rem; margin-bottom:.6rem; }
[data-testid="stTextInput"] input{ direction:rtl; text-align:center;
  font-size:1.05rem; font-weight:500; }

/* הצצה דקורטיבית קלה לצ'אט במסך הפתיחה */
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

/* אנימציית כניסה — כדור מסתובב וגדל */
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
/* כיווניות ימין-לשמאל (RTL) לרשימות, טקסט, שדה הקלט והאקורדיונים */
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
/* בועת הודעה בצ'אט — רקע זכוכית מטושטש עם מסגרת azure */
.stChatMessage{
  background:var(--glass) !important; border:1px solid var(--glass-bd);
  border-radius:14px; backdrop-filter:blur(8px);
}
/* assistant (the agent) bubble — deep blue */
/* בועת העוזר (הסוכן) — כחול עמוק */
.stChatMessage:has(span.ms-bot-tag){
  background:linear-gradient(135deg, rgba(13,30,64,.80), rgba(7,14,30,.72)) !important;
  border-color:rgba(56,214,245,.22) !important; }
/* בועת המשתמש — תכלת בהיר */
.stChatMessage:has(span.ms-user-tag){
  background:linear-gradient(135deg, rgba(16,208,240,.17), rgba(56,214,245,.06)) !important;
  border-color:rgba(56,214,245,.50) !important;
  box-shadow:0 0 0 1px rgba(56,214,245,.12) inset; }
span.ms-bot-tag, span.ms-user-tag{ display:none; }
[data-testid="stSidebar"]{ background:var(--deep); border-inline-end:1px solid var(--glass-bd); }
[data-testid="stSidebar"] *{ direction:rtl; text-align:right; }
/* עיצוב הכפתורים — גלולה שקופה בגוון azure עם מעבר חלק */
.stButton>button{
  background:rgba(16,208,240,.08); color:var(--azure-3);
  border:1px solid var(--glass-bd); border-radius:999px; font-weight:500;
  transition:all .18s ease; font-size:.85rem;
}
.stButton>button:hover{ background:rgba(16,208,240,.18); border-color:var(--azure); color:#fff; }
.stDataFrame thead tr th{ background:var(--elev2) !important; color:var(--azure-3) !important; }
.ms-note{ color:var(--txt3); font-size:.82rem; line-height:1.6; }
hr{ border-color:var(--glass-bd); }

/* ========== אנימציות וליטוש ויזואלי ========== */
@keyframes fadeUp   { from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:none} }
@keyframes glowPulse{ 0%,100%{filter:drop-shadow(0 8px 26px rgba(16,208,240,.35))}
                      50%{filter:drop-shadow(0 12px 44px rgba(16,208,240,.75))} }
@keyframes bgDrift  { 0%{background-position:0% 0%,100% 100%,0 0} 100%{background-position:8% 5%,92% 95%,0 0} }
@keyframes popIn    { 0%{opacity:0; transform:scale(.5)} 70%{transform:scale(1.12)} 100%{opacity:1; transform:scale(1)} }
@keyframes ringspin { to{ transform:rotate(360deg) } }

/* בועות הצ'אט נכנסות בהחלקה ומתרוממות במעבר עכבר */
.stChatMessage{ animation:fadeUp .45s cubic-bezier(.16,1,.3,1) both;
  transition:transform .2s ease, box-shadow .2s ease; }
.stChatMessage:hover{ transform:translateY(-2px);
  box-shadow:0 16px 44px -20px rgba(16,208,240,.5); }

/* רקע "חי" שנע לאט + לוגו פועם */
.stApp{ background-size:140% 140%, 150% 150%, auto;
  animation:bgDrift 20s ease-in-out infinite alternate; }
.ms-logo{ animation:glowPulse 3.4s ease-in-out infinite; }

/* פס גלילה מותאם בגוון azure */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:linear-gradient(var(--azure-deep),var(--azure));
  border-radius:10px; border:2px solid var(--abyss); }
::-webkit-scrollbar-thumb:hover{ background:var(--azure-2); }

/* מיקרו-אינטראקציות לטבלאות ולכפתורים */
.stDataFrame{ transition:transform .2s ease, box-shadow .2s ease; border-radius:10px; }
.stDataFrame:hover{ transform:translateY(-2px);
  box-shadow:0 16px 44px -22px rgba(16,208,240,.45); }
.stButton>button:active{ transform:scale(.96); }

/* ---- כרטיס שחקן משודרג (כותרת + תמונה + דירוג) ---- */
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
/* ---- כרטיס שחקן בסגנון FIFA (FUT) ---- */
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

/* מקרא העמודות המוצג מתחת לכל טבלה */
.ms-legend{ direction:rtl; text-align:right; font-size:.8rem; color:var(--txt2);
  line-height:2; margin:.45rem 0 .2rem; padding:.55rem .85rem;
  background:rgba(16,208,240,.05); border:1px solid var(--glass-bd);
  border-radius:10px; }
.ms-legend code{ color:var(--azure-3); font-family:'Fira Code', monospace;
  font-size:.74rem; background:rgba(16,208,240,.10); padding:1px 6px;
  border-radius:5px; direction:ltr; display:inline-block; }
/* כותרת ההשוואה הוויזואלית בין שחקנים דומים */
.ms-cmp-title{ direction:rtl; text-align:center; color:var(--txt); font-weight:700;
  font-size:1.02rem; margin:.7rem 0 .35rem; }
.ms-cmp-title b{ color:var(--azure-3); }

@media (prefers-reduced-motion: reduce){ *{ animation:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# מקודדים את הלוגו ל-base64 כדי להטמיע אותו ישירות ב-HTML
_LOGO_B64 = base64.b64encode((ROOT / "assets" / "logo.png").read_bytes()).decode()
# מציגים את הלוגו, שורת התיאור וקרדיט הזכויות בראש הדף
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
# שער השם — אם עוד אין שם משתמש בסשן
if "username" not in st.session_state:
    # entrance animation: a spinning, growing ball, then into the chat
    # אנימציית כניסה — אם המשתמש בדיוק הזין שם
    if st.session_state.get("entering"):
        # השם שהוזן
        _name = st.session_state.entering
        # מציגים את אנימציית הכניסה עם ברכה
        st.markdown(f"<div class='ms-enter'><div class='ms-enter-ball'>⚽</div>"
                    f"<div class='ms-enter-text'>ברוכים הבאים, {_name}! 🚀</div></div>",
                    unsafe_allow_html=True)
        # מייבאים time להשהיה קצרה
        import time as _t
        # משהים כדי שהאנימציה תיראה
        _t.sleep(1.4)
        # שומרים את השם בסשן
        st.session_state.username = _name
        # מוחקים את דגל הכניסה
        del st.session_state["entering"]
        # מרעננים את הדף כדי להיכנס לצ'אט
        st.rerun()

    # מציגים את כותרת שער השם
    st.markdown("<div class='ms-gate'><h3>ברוכים הבאים 👋</h3>"
                "<p>איך לקרוא לך? כך אוכל לפנות אליך באופן אישי.</p></div>",
                unsafe_allow_html=True)
    # יוצרים עמודה מרכזית לטופס
    _c = st.columns([1, 2, 1])[1]
    # בתוך העמודה המרכזית
    with _c:
        # טופס קלט השם
        with st.form("name_gate", border=False):
            # שדה הזנת השם
            _nm = st.text_input("שם", label_visibility="collapsed", placeholder="הקלידו שם…")
            # כפתור השליחה
            _go = st.form_submit_button("כניסה ⚽", use_container_width=True)
        # אם נלחץ הכפתור ויש שם — שומרים אותו ומרעננים
        if _go and _nm.strip():
            # שומרים את השם (עד 24 תווים) בדגל הכניסה
            st.session_state.entering = _nm.strip()[:24]
            # מרעננים
            st.rerun()
    # a light, decorative peek at the conversation
    # הצצה דקורטיבית קלה לשיחה
    st.markdown(
        "<div class='ms-preview'>"
        "<div class='ms-pv-row'><span class='ms-pv user'>מצא חלוצים מהירים מתחת לגיל 23 💨</span></div>"
        "<div class='ms-pv-row'><span class='ms-pv bot'>הנה 3 חלוצים מהירים ומבטיחים בתקציב שלך ⚽</span></div>"
        "<div class='ms-pv-row'><span class='ms-pv user'>מי דומה ל-Messi? 🐐</span></div>"
        "<div class='ms-pv-row'><span class='ms-pv bot'>הדומים ביותר בסגנון: Dybala, Martial, Ben Yedder 📊</span></div>"
        "</div>", unsafe_allow_html=True)
    # עוצרים כאן עד שיוזן שם (לא מציגים את הצ'אט)
    st.stop()

# make the name available to the CSS (shown under the user avatar)
# מנקים את שם המשתמש (הסרת תווים מסוכנים) ומעבירים אותו ל-CSS
_uname = re.sub(r'["<>]', "", st.session_state.username)
# מזריקים את השם כמשתנה CSS (יוצג מתחת לאווטר המשתמש)
st.markdown(f"<style>:root{{--ms-username:\"{_uname}\"}}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data + key
# ---------------------------------------------------------------------------
# טוענים את הטבלה המרכזית (עם מטמון כדי לא לטעון בכל ריצה)
@st.cache_data(show_spinner=False)
def get_table():
    # מחזירים את הטבלה מהסוכן
    return agent.load_table()


# בודקים אם מפתח ה-API מוכן (האם ניתן לבנות לקוח)
def key_ready() -> bool:
    # מנסים לבנות את הלקוח
    try:
        agent._get_client()
        # הצליח — המפתח מוכן
        return True
    except Exception:
        # נכשל — אין מפתח
        return False


# טוענים את הטבלה פעם אחת
df = get_table()


# מחזיר שורה מלאה (כל העמודות) של שחקן לפי שמו הקצר — לכרטיס/רדאר
def get_full_row(name):
    """Full-column row for a player by short_name (for the card / radar)."""
    # מסננים לפי השם הקצר
    m = df[df["short_name"].astype(str) == str(name)]
    # מחזירים את ההתאמה בעלת הדירוג הגבוה ביותר, או None
    return m.sort_values("overall", ascending=False).iloc[0] if len(m) else None


# ---------------------------------------------------------------------------
# Charts + player card
# ---------------------------------------------------------------------------
# פלטת הצבעים לגרפים (azure וגוונים משלימים)
_PALETTE = ["#10d0f0", "#f5c26b", "#9fe7c0", "#e0729f", "#7fe6fb", "#b39ddb"]


# פונקציה: יוצרת גרף פיזור דו-ממדי (PCA) של הקלאסטרים, צבע לכל סגנון
def make_cluster_scatter(xy, cluster_ids, label_map):
    """2D PCA scatter of the clusters (azure palette, one colour per style)."""
    # יוצרים דמות וציר
    fig, ax = plt.subplots(figsize=(7, 5))
    # רקע שקוף לדמות
    fig.patch.set_alpha(0)
    # רקע שקוף לציר
    ax.set_facecolor("none")
    # מציירים נקודות לכל קלאסטר בצבע משלו
    for i, cid in enumerate(sorted(set(cluster_ids))):
        # מסכה לחברי הקלאסטר הנוכחי
        m = cluster_ids == cid
        # פיזור הנקודות עם תווית הסגנון
        ax.scatter(xy[m, 0], xy[m, 1], s=15, alpha=.7, edgecolors="none",
                   color=_PALETTE[i % len(_PALETTE)], label=label_map.get(cid, str(cid)))
    # תווית ציר X (רכיב ראשי 1)
    ax.set_xlabel("Principal Component 1", color="#c5ccda", fontsize=9)
    # תווית ציר Y (רכיב ראשי 2)
    ax.set_ylabel("Principal Component 2", color="#c5ccda", fontsize=9)
    # צבע סימוני הצירים
    ax.tick_params(colors="#8a94ab", labelsize=8)
    # צובעים את מסגרות הציר
    for spine in ax.spines.values():
        spine.set_color("#26324a")
    # מקרא הסגנונות
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor="#c5ccda")
    # פריסה מהודקת
    fig.tight_layout()
    # מחזירים את הדמות
    return fig


# 6 התכונות בצירי הרדאר
_RADAR_ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
# תוויות הצירים באנגלית
_RADAR_LABELS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]


# פונקציה: יוצרת גרף רדאר של 6 תכונות הליבה לשחקן אחד או יותר
def make_radar(players, colors=None):
    """Radar/spider chart of the 6 core attributes for one or more players.
    `colors` optionally overrides the per-player line colour (e.g. azure vs red
    for the similar-player comparison)."""
    # מחשבים את הזוויות של צירי הרדאר
    angles = np.linspace(0, 2 * np.pi, len(_RADAR_ATTRS), endpoint=False).tolist()
    # סוגרים את המעגל (מוסיפים את הזווית הראשונה בסוף)
    angles += angles[:1]
    # יוצרים דמות וציר קוטבי
    fig, ax = plt.subplots(figsize=(4.4, 4.4), subplot_kw=dict(polar=True))
    # רקע שקוף לדמות
    fig.patch.set_alpha(0)
    # רקע שקוף לציר
    ax.set_facecolor("none")
    # מציירים קו לכל שחקן
    for i, p in enumerate(players):
        # ערכי 6 התכונות של השחקן
        vals = [float(p[a]) for a in _RADAR_ATTRS]
        # סוגרים את המעגל
        vals += vals[:1]
        # צבע הקו (מהפרמטר או מהפלטה)
        c = colors[i] if (colors and i < len(colors)) else _PALETTE[i % len(_PALETTE)]
        # מציירים את קו הרדאר
        ax.plot(angles, vals, color=c, linewidth=2, label=str(p["short_name"]))
        # ממלאים את השטח מתחת לקו
        ax.fill(angles, vals, color=c, alpha=.22)
        # value number at each vertex (single-player card only — avoids clutter)
        # מספר הערך בכל קודקוד (רק לכרטיס בודד — למניעת עומס)
        if len(players) == 1:
            # עוברים על כל זווית וערך
            for ang, v in zip(angles[:-1], vals[:-1]):
                # מציירים את ערך התכונה ליד הקודקוד
                ax.annotate(str(int(round(v))), xy=(ang, min(v + 7, 99)),
                            ha="center", va="center", color="#7fe6fb",
                            fontsize=10, fontweight="bold")
    # מציבים את סימוני הזוויות
    ax.set_xticks(angles[:-1])
    # תוויות התכונות
    ax.set_xticklabels(_RADAR_LABELS, color="#c5ccda", fontsize=9)
    # סימוני הסקאלה הרדיאלית
    ax.set_yticks([20, 40, 60, 80, 100])
    # תוויות הסקאלה
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#5e6b85", fontsize=7)
    # טווח הסקאלה 0–100
    ax.set_ylim(0, 100)
    # צבע קווי הרשת
    ax.grid(color="#26324a")
    # צבע מסגרת הרדאר
    ax.spines["polar"].set_color("#26324a")
    # אם יש יותר משחקן אחד — מוסיפים מקרא
    if len(players) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8,
                  frameon=False, labelcolor="#c5ccda")
    # פריסה מהודקת
    fig.tight_layout()
    # מחזירים את הדמות
    return fig


# מביא תמונה + דגל לשחקן מ-API של EA (גם לשחקנים שבמאגר), עם מטמון
@st.cache_data(show_spinner=False)
def enrich_media(name):
    """Photo + national flag for a player, fetched live from EA's ratings API
    (so even players that ARE in our data get a portrait + flag). Cached."""
    # מנסים לשלוף מדיה מ-EA
    try:
        return external.ea_media(name)
    except Exception:
        # אם נכשל — מילון ריק
        return {}


# פונקציית עזר: מחזירה ערך שדה אם קיים ואינו NaN
def _val(p, key):
    # שולפים את הערך (תומך Series/dict)
    v = p.get(key) if hasattr(p, "get") else None
    # מחזירים רק אם קיים ולא NaN
    return v if (v is not None and pd.notna(v)) else None


# פונקציית עזר: קובעת את דרגת הכרטיס (זהב/כסף/ארד) לפי הדירוג
def _tier(v):
    # ללא דירוג או 80+ → זהב (אם אין דירוג → כסף)
    if v is None or v >= 80:
        return "gold" if v is not None else "silver"
    # 70–79 → כסף, אחרת ארד
    return "silver" if v >= 70 else "bronze"


# מיפוי קבוצת עמדה לקיצור המוצג בכרטיס
_POS_ABBR = {"Forward": "FWD", "Midfielder": "MID", "Defender": "DEF", "GK": "GK"}
# 6 תכונות הכרטיס: (קיצור תצוגה, שם העמודה)
_FUT_STATS = [("PAC", "pace"), ("SHO", "shooting"), ("PAS", "passing"),
              ("DRI", "dribbling"), ("DEF", "defending"), ("PHY", "physic")]


# פונקציה: בונה את ה-HTML של כרטיס השחקן בסגנון FUT (זהב/כסף/ארד)
def _player_card_html(p):
    """Build the FIFA-style (FUT) card HTML — gold (80+) / silver (70-79) /
    bronze (<70) — photo, OVR, position, 6 stats and the national flag.
    Photo/flag enriched live from EA for players in our data."""
    # שם השחקן
    name = str(p["short_name"])
    # כתובת התמונה (אם קיימת)
    av = _val(p, "avatar_url")
    # כתובת הדגל (אם קיימת)
    flag = _val(p, "nationality_flag_url")
    # אם חסרה תמונה או דגל — משלימים מ-EA
    if av is None or flag is None:
        # שולפים מדיה מ-EA
        m = enrich_media(name)
        # ממלאים תמונה אם חסרה
        av = av or m.get("avatar_url")
        # ממלאים דגל אם חסר
        flag = flag or m.get("flag_url")
    # הדירוג הכללי כמספר שלם
    ovr = int(p["overall"]) if pd.notna(p.get("overall")) else None
    # דרגת הכרטיס לפי הדירוג
    tier = _tier(ovr)
    # קיצור העמדה
    pos = _POS_ABBR.get(p.get("position_group"), p.get("position_group") or "")
    # שם המשפחה (הסרת ראשי תיבה מהשם הקצר)
    last = re.sub(r"^[A-Z]\.\s*", "", name).strip() or name

    # פונקציה פנימית: מחזירה ערך תכונה כמספר שלם או מקף
    def s(k):
        # ערך התכונה
        v = p.get(k)
        # ממירים ל-int או מקף אם חסר
        return int(v) if (v is not None and pd.notna(v)) else "—"
    # בונים את HTML של 6 התכונות
    stats_html = "".join(f"<div><b>{s(k)}</b><span>{lbl}</span></div>" for lbl, k in _FUT_STATS)
    # HTML של התמונה (או אמוג'י כדור אם אין)
    photo_html = f"<img src='{av}'>" if av else "<div class='fut-noimg'>⚽</div>"
    # HTML של הדגל (אם קיים)
    flag_html = f"<div class='fut-bottom'><img src='{flag}'></div>" if flag else ""
    # מחזירים את ה-HTML המלא של הכרטיס
    return (f"<div class='fut-card fut-{tier}'>"
            f"<div class='fut-top'><div class='fut-ovr'>{ovr if ovr is not None else '—'}</div>"
            f"<div class='fut-pos'>{pos}</div></div>"
            f"<div class='fut-photo'>{photo_html}</div>"
            f"<div class='fut-name'>{last}</div>"
            f"<div class='fut-stats'>{stats_html}</div>{flag_html}</div>")


# פונקציה: בונה את שורת הפרטים הקצרה שמוצגת מתחת לכרטיס
def _facts_line(p):
    """The short textual details shown beneath a card."""
    # רשימת הפרטים
    facts = []
    # גיל
    if pd.notna(p.get("age")):
        facts.append(f"גיל {int(p['age'])}")
    # פוטנציאל
    if pd.notna(p.get("potential")):
        facts.append(f"פוטנציאל {int(p['potential'])}")
    # שווי שוק
    if pd.notna(p.get("value_eur")):
        facts.append(f"שווי €{int(p['value_eur']):,}")
    # רגל מועדפת
    if p.get("preferred_foot"):
        facts.append(f"רגל {p['preferred_foot']}")
    # לאום
    if p.get("nationality_name"):
        facts.append(str(p["nationality_name"]))
    # אם יש נתוני אירועים — מוסיפים גולים ודאבלים
    if p.get("has_event_data"):
        facts.append(f"⚽ {int(p.get('total_goals', 0))} גולים")
        facts.append(f"דאבלים {int(p.get('matches_with_2_plus_goals', 0))}")
    # מחברים את הפרטים במפריד
    return " · ".join(facts)


# the 6 FUT-card stat abbreviations + OVR, explained in Hebrew (the card legend)
# מקרא הכרטיס: קיצורי 6 התכונות + OVR והסברם בעברית
_FUT_STAT_LEGEND = [
    ("OVR", "דירוג כללי (0–99)"), ("PAC", "מהירות"), ("SHO", "בעיטות / גמר"),
    ("PAS", "מסירות"), ("DRI", "כדרור"), ("DEF", "הגנה"), ("PHY", "פיזיות / חוזק"),
]


# פונקציה: מציגה את מקרא קיצורי הכרטיס בעברית
def render_card_legend():
    """Explain the player card's stat abbreviations (PAC/SHO/…) in plain Hebrew."""
    # בונים את פריטי המקרא
    items = " &nbsp;·&nbsp; ".join(f"<code>{a}</code> {h}" for a, h in _FUT_STAT_LEGEND)
    # מציגים את המקרא
    st.markdown(f"<div class='ms-legend'>📖 <b>מקרא הכרטיס:</b><br>{items}</div>",
                unsafe_allow_html=True)


# פונקציה: מציגה כרטיס שחקן בודד — רדאר לצד כרטיס FUT, פרטים ומקרא מתחת
def render_player_card(p):
    """A single player: compact radar next to the FUT card, facts + legend beneath."""
    # שתי עמודות: רדאר וכרטיס
    c1, c2 = st.columns([1, 1])
    # עמודה ראשונה — הרדאר
    with c1:
        st.pyplot(make_radar([p]), use_container_width=True)
    # עמודה שנייה — כרטיס ה-FUT
    with c2:
        st.markdown(_player_card_html(p), unsafe_allow_html=True)
    # שורת הפרטים מתחת
    st.markdown(f"<div class='fut-facts'>{_facts_line(p)}</div>", unsafe_allow_html=True)
    # מקרא הכרטיס
    render_card_legend()


# פונקציית עזר: True רק אם לשחקן יש כל 6 התכונות לרדאר
def _has_radar(p):
    """True only if the player has all 6 numeric face attributes for the radar."""
    # בודקים שהשחקן קיים ושכל התכונות אינן NaN
    return p is not None and all(pd.notna(_val(p, a)) for a in _RADAR_ATTRS)


# פונקציה: תצוגת ההשוואה "דומה ל-X" — היעד וההתאמה הקרובה על רדאר אחד + שני כרטיסים
def render_similar_compare(art, dfa):
    """The 'find similar to X' hero view: the target player and the single closest
    match overlaid on ONE radar (azure vs red) + their two cards side by side.
    Works whether the target is in our table OR came from EA / the model (its 6
    attributes are carried in art['target_row']). Returns True if it rendered."""
    # שם שחקן היעד
    target_name = art.get("target")
    # אם אין יעד או אין תוצאות — לא מציגים
    if not target_name or dfa is None or dfa.empty:
        return False
    # target: prefer our table; otherwise rebuild it from the external attributes
    # שורת היעד: מעדיפים מהמאגר; אחרת בונים מהתכונות החיצוניות
    target = get_full_row(str(target_name))
    # אם אין במאגר אבל יש שורת תכונות חיצונית — משתמשים בה
    if target is None and art.get("target_row"):
        target = pd.Series(art["target_row"])
    # שם ההתאמה הקרובה ביותר
    match_name = str(dfa.iloc[0]["short_name"])
    # שורת ההתאמה המלאה
    match = get_full_row(match_name)
    # אם לאחד מהם חסרות תכונות לרדאר — נופלים לתצוגת טבלה בלבד
    if not _has_radar(target) or not _has_radar(match):
        return False  # missing attributes — fall back to the table-only view

    # ערך הדמיון של ההתאמה
    sim = dfa.iloc[0].get("similarity")
    # טקסט אחוז הדמיון (אם קיים)
    sim_txt = f" · דמיון {float(sim) * 100:.0f}%" if pd.notna(sim) else ""
    # כותרת ההשוואה הוויזואלית
    st.markdown(
        f"<div class='ms-cmp-title'>🔬 השוואה ויזואלית: <b>{target_name}</b> "
        f"מול ההתאמה הקרובה ביותר <b>{match_name}</b>{sim_txt}</div>",
        unsafe_allow_html=True)
    # the target wasn't in our table — its 6 attributes came from an external source
    # אם היעד אינו במאגר — מציינים מאיפה נשלפו תכונותיו
    if art.get("source") in ("ea", "web"):
        # ניסוח המקור (EA או ידע המודל)
        whence = ("הדירוגים הרשמיים של EA Sports FC" if art["source"] == "ea"
                  else "ידע המודל על השחקן")
        # הערת מקור
        st.markdown(f"<div class='ms-note' style='text-align:center'>ℹ️ <b>{target_name}</b> "
                    f"אינו במאגרים שלנו — 6 התכונות שלו נשלפו מ{whence}.</div>",
                    unsafe_allow_html=True)

    # עמודה מרכזית לרדאר
    mid = st.columns([1, 2, 1])[1]
    # מציירים את הרדאר המולבש (כחול ליעד, אדום להתאמה)
    with mid:
        st.pyplot(make_radar([target, match], colors=["#10d0f0", "#ff4d5e"]),
                  use_container_width=True)

    # שתי עמודות לשני הכרטיסים
    c1, c2 = st.columns(2)
    # כרטיס היעד
    with c1:
        st.markdown(_player_card_html(target), unsafe_allow_html=True)
        st.markdown(f"<div class='fut-facts'>{_facts_line(target)}</div>",
                    unsafe_allow_html=True)
    # כרטיס ההתאמה
    with c2:
        st.markdown(_player_card_html(match), unsafe_allow_html=True)
        st.markdown(f"<div class='fut-facts'>{_facts_line(match)}</div>",
                    unsafe_allow_html=True)
    # מקרא הצבעים (כחול=היעד, אדום=ההתאמה)
    st.markdown(
        "<div class='ms-note' style='text-align:center'>🔵 הקו הכחול = השחקן שביקשת"
        f" (<b>{target_name}</b>) · 🔴 הקו האדום = ההתאמה (<b>{match_name}</b>). "
        "ככל שהצורות חופפות יותר — סגנון המשחק דומה יותר.</div>",
        unsafe_allow_html=True)
    # מקרא הכרטיס
    render_card_legend()
    # מסמנים שהתצוגה רצה
    return True


# פונקציה: פעולת הבורר — לשחקן הנבחר מציגים השוואה דינמית מול הדומה לו ביותר
def render_pick_comparison(name):
    """Picker action: for the chosen player, find their single closest match and
    show the dynamic comparison (overlaid radar + both cards). Recomputed every
    time the selection changes. Falls back to a single card if no comparison can
    be built (e.g. a goalkeeper, or no candidates)."""
    # מנסים לחשב דמיון ולהציג השוואה
    try:
        # מריצים דמיון לשחקן הנבחר
        res, extra = agent.route_query(
            "similar_players", {"player_name": name, "top_n": 4}, df)
        # בונים אובייקט השוואה
        cmp_art = {"name": "find_similar_players", "target": name, "source": "primary"}
        # אם ההשוואה הוצגה — מסיימים
        if render_similar_compare(cmp_art, res):
            return
    except Exception:
        # כשל (למשל שוער) — ממשיכים לנפילת החן
        pass
    # נפילת חן: מציגים כרטיס בודד של השחקן
    row = get_full_row(name)
    # אם נמצאה שורה — מציגים כרטיס
    if row is not None:
        render_player_card(row)


# ---------------------------------------------------------------------------
# Column legend (מקרא) — a short Hebrew explainer shown beneath every table
# ---------------------------------------------------------------------------
# מילון מקרא: שם עמודה → הסבר קצר בעברית (מוצג מתחת לכל טבלה)
COL_LEGEND_HE = {
    "short_name": "שם השחקן",
    "long_name": "שם מלא",
    "position_group": "עמדה (חלוץ/קשר/הגנה/שוער)",
    "age": "גיל",
    "overall": "דירוג כללי ב-FC24 (0–99)",
    "potential": "פוטנציאל עתידי (0–99)",
    "value_eur": "שווי שוק ביורו (€)",
    "preferred_foot": "רגל חזקה (ימין/שמאל)",
    "club_name": "מועדון",
    "league_name": "ליגה",
    "nationality_name": "נבחרת / לאום",
    "matches": "מספר משחקים (מנתוני אירועים אמיתיים)",
    "total_goals": "סך הגולים שהבקיע",
    "goals_per_match": "ממוצע גולים למשחק",
    "matches_with_2_plus_goals": "משחקי דאבל (2+ גולים באותו משחק)",
    "attacking_involvement_score": "מעורבות התקפית (0–100, גבוה=מעורב יותר)",
    "creative_score": "יצירתיות ובישולים (0–100)",
    "discipline_score": "משמעת — מעט כרטיסים (0–100, גבוה=ממושמע)",
    "foot_balance_score": "איזון דו-רגלי (0–100, גבוה=שולט בשתי הרגליים)",
    "similarity": "מידת דמיון לשחקן המבוקש (0–1, כאשר 1=זהה לחלוטין)",
    "reason": "הסבר הדמיון — התכונות שבהן השחקנים הכי קרובים",
    "cluster_id": "מספר מזהה של קבוצת הסגנון",
    "label": "שם סגנון המשחק שזוהה",
    "size": "כמות השחקנים בקבוצה",
    "market_efficiency_score": "יחס יכולת-למחיר (גבוה=מציאה משתלמת יותר)",
    "anomaly_score": "ציון חריגות (נמוך=יוצא דופן יותר)",
    "direction": "סוג החריגה (מבצע יתר / מתחת לציפיות)",
    "conversion_rate": "אחוז המרת בעיטות לגולים",
}
# מקרא דינמי: קידומות עמודות הקלאסטרים (trait_/player_) להסבר בעברית
_LEGEND_DYNAMIC = {"trait_": "תכונה דומיננטית", "player_": "שחקן לדוגמה"}


# פונקציה: מחזירה את תווית המקרא בעברית לעמודה נתונה (או None אם אין)
def _col_label(c):
    # אם העמודה במילון הקבוע — מחזירים את ההסבר
    if c in COL_LEGEND_HE:
        return COL_LEGEND_HE[c]
    # אחרת בודקים קידומות דינמיות (trait_1, player_2 וכו')
    for pre, lab in _LEGEND_DYNAMIC.items():
        # אם העמודה מתחילה בקידומת ומסתיימת במספר
        if c.startswith(pre) and c[len(pre):].isdigit():
            return f"{lab} #{c[len(pre):]}"
    # לא נמצא הסבר
    return None


# פונקציה: מציגה מתחת לטבלה מקרא של כל עמודה בעברית
def render_column_legend(dfa):
    """List, beneath a table, what each of its columns means — in plain Hebrew."""
    # בונים פריט מקרא לכל עמודה שיש לה הסבר
    items = [f"<code>{c}</code> {_col_label(c)}"
             for c in dfa.columns if _col_label(c)]
    # אם אין פריטים — לא מציגים
    if not items:
        return
    # מציגים את המקרא
    st.markdown("<div class='ms-legend'>📖 <b>מקרא עמודות:</b><br>"
                + " &nbsp;·&nbsp; ".join(items) + "</div>",
                unsafe_allow_html=True)


# compact "about / sources / ethics" — collapsed, so the top stays clean
# בלוק "אודות / מקורות / אתיקה" מכווץ — כדי שראש הדף יישאר נקי
with st.expander("ℹ️ אודות · מקורות נתונים · אתיקה"):
    # מציגים את פרטי הנתונים, סדר המקורות, הגבול והאתיקה
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
# אם אין עדיין היסטוריית שיחה — מאתחלים אותה
if "history" not in st.session_state:
    # בונים ברכת פתיחה אישית עם שם המשתמש
    _greet = agent.greeting_message("he").replace("היי!", f"היי {_uname}!", 1)
    # היסטוריית התצוגה מתחילה מהברכה
    st.session_state.history = [{"role": "assistant", "content": _greet}]
    # היסטוריית ה-LLM מתחילה בהודעת מערכת עם שם המשתמש
    st.session_state.llm = [{"role": "system",
        "content": f"The user's name is {_uname}. Address them warmly by name now "
                   f"and then (especially when greeting), in their language."}]

# אם אין מפתח API — מציגים אזהרה ידידותית
if not key_ready():
    st.warning("מפתח OpenAI לא נמצא. מקומית: הוסיפו ל-`.env`. "
               "ב-Streamlit Cloud: Settings → Secrets → `OPENAI_API_KEY`.")

# כלים שמחזירים רשימת שחקנים (לזיהוי היכן ממקמים את בורר ההשוואה)
_PLAYER_TOOLS = ("search_players", "find_similar_players", "detect_bargains")


# תוויות מקור הנתונים לתצוגה בתחתית כל תוצאה
_SRC_LABEL = {"primary": "מאגר ראשי (FC24) + משני (אירועים)",
              "ea": "EA Sports FC — דירוגים רשמיים (API חי)",
              "web": "ידע המודל (חיפוש בדומיין)"}


# פונקציה: מציגה כותרת תחתית עם מקור הנתונים ומספר השורות
def _src_footer(src, n=None):
    # תווית המקור
    lbl = _SRC_LABEL.get(src, "")
    # קידומת מספר השורות (אם ניתן)
    pre = f"📋 {n} שורות · " if n is not None else ""
    # אם יש תווית מקור — מציגים אותה עם הקידומת
    if lbl:
        st.markdown(f"<span class='ms-note'>{pre}📊 מבוסס על: {lbl}</span>",
                    unsafe_allow_html=True)
    # אחרת אם יש רק קידומת — מציגים אותה
    elif pre:
        st.markdown(f"<span class='ms-note'>{pre.rstrip(' ··')}</span>",
                    unsafe_allow_html=True)


# פונקציה: מציגה את ה-artifact (תוצאה ויזואלית) לפי סוג הכלי שרץ
def render_artifact(art):
    # שם הכלי, טבלת התוצאה ומקור הנתונים
    name, dfa, src = art["name"], art["df"], art.get("source")
    # one player card (not a disambiguation list)
    # מקרה כרטיס שחקן בודד (לא רשימת הבחנה)
    if name == "player_profile" and not art.get("disambig"):
        # אם המקור EA — הערה שהנתונים נשלפו בזמן אמת מ-EA
        if src == "ea":
            st.info("ℹ️ שחקן זה אינו במאגרים שלנו — הנתונים והתמונה נשלפו **בזמן אמת** "
                    "מהדירוגים הרשמיים של EA Sports FC (ea.com/ratings).")
        # אם המקור ידע המודל — הערה בהתאם
        elif src == "web":
            st.info("⚠️ שחקן זה אינו במאגרים שלנו ולא נמצא ב-EA — הכרטיס נבנה בעזרת "
                    "המודל לפי ידע על השחקן (חיפוש בדומיין).")
        # מציגים את כרטיס השחקן
        render_player_card(dfa.iloc[0])
        # כותרת תחתית עם המקור
        _src_footer(src)
        # מסיימים
        return
    # מקרה קיבוץ — מציגים טבלה, מקרא וגרף פיזור
    if name == "cluster_players":
        # טבלת הקלאסטרים
        st.dataframe(dfa, use_container_width=True, hide_index=True)
        # מקרא העמודות
        render_column_legend(dfa)
        # נתוני הפיזור (אם קיימים)
        sc = art.get("scatter")
        # אם יש — מציירים את גרף הפיזור
        if sc:
            st.pyplot(make_cluster_scatter(np.array(sc["xy"]), np.array(sc["cids"]),
                                           sc["label_map"]), use_container_width=True)
        # כותרת תחתית עם המקור
        _src_footer(src)
        # מסיימים
        return
    # "find similar to X" -> overlay the target and the closest match (azure vs red)
    # on one radar + show both cards, then the full ranked table below.
    # מקרה "דומה ל-X" — מציגים את ההשוואה המולבשת מעל הטבלה
    if name == "find_similar_players" and not art.get("disambig"):
        render_similar_compare(art, dfa)
    # a table: players list OR disambiguation candidates
    # מציגים טבלה — רשימת שחקנים או מועמדי הבחנה
    st.dataframe(dfa, use_container_width=True, hide_index=True)
    # אם זו רשימת הבחנה — מבקשים מהמשתמש לכתוב את השם המלא
    if art.get("disambig"):
        st.markdown("<span class='ms-note'>נמצאו כמה שחקנים בשם הזה — כתבו את "
                    "השם המלא של זה שמעניין אתכם.</span>", unsafe_allow_html=True)
    # מקרא העמודות
    render_column_legend(dfa)
    # כותרת תחתית עם המקור ומספר השורות
    _src_footer(src, len(dfa))


# index of the most recent players table -> the inline card picker sits there
# מאתרים את האינדקס של טבלת השחקנים האחרונה (שם ימוקם בורר ההשוואה)
_last_pidx = None
# עוברים על כל ההיסטוריה
for _i, _m in enumerate(st.session_state.history):
    # ה-artifact של ההודעה (אם יש)
    _a = _m.get("artifact")
    # אם זו טבלת שחקנים תקינה — מעדכנים את האינדקס האחרון
    if (_a and _a["name"] in _PLAYER_TOOLS and not _a.get("disambig")
            and "short_name" in _a["df"].columns):
        _last_pidx = _i

# render the conversation so far (with the card picker inline at the latest result)
# מציגים את כל השיחה עד כה (עם בורר ההשוואה בתוצאה האחרונה)
for _i, m in enumerate(st.session_state.history):
    # בועת הצ'אט לפי תפקיד (עם האווטר המתאים)
    with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
        # תגית CSS להבחנת בועת עוזר/משתמש
        _tag = "ms-bot-tag" if m["role"] == "assistant" else "ms-user-tag"
        # מזריקים את התגית
        st.markdown(f"<span class='{_tag}'></span>", unsafe_allow_html=True)
        # אם יש תוכן טקסט — מציגים אותו
        if m.get("content"):
            st.markdown(m["content"])
        # אם יש artifact — מציגים אותו
        if m.get("artifact"):
            render_artifact(m["artifact"])
        # אם זו התוצאה האחרונה — מוסיפים את בורר ההשוואה
        if _i == _last_pidx:
            # ה-artifact הנוכחי
            _art0 = m["artifact"]
            # רשימת שמות לבורר (עם אפשרות ריקה)
            _names = ["—"] + _art0["df"]["short_name"].astype(str).tolist()
            # if this list came from a "find similar to X" query, keep X fixed and
            # let the dropdown choose WHICH of the similar players to compare to X.
            # האם זו רשימת "דומה ל-X" (אז X קבוע והבורר בוחר מי להציג מולו)
            _is_sim = _art0["name"] == "find_similar_players" and _art0.get("target")
            # תווית הבורר במקרה "דומה ל-X"
            if _is_sim:
                _label = (f"🔎 השוואה מול {_art0['target']} — בחרו מהרשימה את השחקן "
                          f"להציג מולו")
            else:
                # תווית הבורר במקרה רגיל
                _label = ("🔎 השוואת דמיון — בחרו שחקן מהרשימה והוא יושווה אוטומטית "
                          "לדומה לו ביותר")
            # תיבת הבחירה
            _pick = st.selectbox(_label, _names, key=f"pick_{_i}")
            # אם נבחר שחקן (לא הערך הריק)
            if _pick != "—":
                # מקרה "דומה ל-X": משווים את הנבחר מול היעד הקבוע
                if _is_sim:
                    # שורת השחקן הנבחר מתוך הרשימה
                    _picked_df = _art0["df"][
                        _art0["df"]["short_name"].astype(str) == _pick
                    ].reset_index(drop=True)
                    # בונים אובייקט השוואה עם היעד הקבוע
                    _cmp_art = {"name": "find_similar_players",
                                "target": _art0["target"],
                                "target_row": _art0.get("target_row"),
                                "source": _art0.get("source")}
                    # מציגים את ההשוואה; אם נכשל — נופלים להשוואה הדינמית
                    if not render_similar_compare(_cmp_art, _picked_df):
                        render_pick_comparison(_pick)
                else:
                    # מקרה רגיל: השוואה דינמית של הנבחר מול הדומה לו ביותר
                    render_pick_comparison(_pick)


# ---------------------------------------------------------------------------
# Chat input -> one conversational turn
# ---------------------------------------------------------------------------
# שדה קלט הצ'אט — כל הקלדה מפעילה תור שיחה אחד
if prompt := st.chat_input("כתבו לסוכן בשפה חופשית…"):
    # מוסיפים את הודעת המשתמש להיסטוריית התצוגה
    st.session_state.history.append({"role": "user", "content": prompt})
    # ומוסיפים אותה גם להיסטוריית ה-LLM
    st.session_state.llm.append({"role": "user", "content": prompt})
    # מציגים מיד את בועת המשתמש
    with st.chat_message("user", avatar=AVATARS["user"]):
        # תגית בועת המשתמש
        st.markdown("<span class='ms-user-tag'></span>", unsafe_allow_html=True)
        # תוכן ההודעה
        st.markdown(prompt)
    # בועת העוזר עם ספינר "חושב…"
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        # תגית בועת העוזר
        st.markdown("<span class='ms-bot-tag'></span>", unsafe_allow_html=True)
        # ספינר בזמן החשיבה
        with st.spinner("חושב…"):
            # מנסים להריץ תור שיחה
            try:
                text, action = agent.converse(st.session_state.llm, df)
            except Exception as e:
                # אם יש שגיאה — מודיעים עליה בלי לקרוס
                text, action = f"מצטער, נתקלתי בשגיאה: {e}", None

    # אתחול ה-artifact שיוצג (אם רצה כלי)
    art = None
    # אם רצה כלי בתור הזה
    if action:
        # המידע הנוסף מהפעולה
        ex = action.get("extra", {}) or {}
        # מקור הנתונים
        src = ex.get("source")
        # מקרה קיבוץ — מנסים להוסיף נתוני פיזור לגרף
        if action["name"] == "cluster_players":
            # מנסים לחשב את נקודות הפיזור
            try:
                # קואורדינטות ותוויות הקלאסטרים
                xy, cids = clustering.cluster_xy(ex["labeled"])
                # מיפוי מזהה קלאסטר → תווית
                lm = {d["cluster_id"]: d["label"] for d in ex["descriptions"]}
                # בונים artifact עם נתוני הפיזור
                art = {"name": "cluster_players", "df": action["df"], "source": src,
                       "scatter": {"xy": xy.tolist(), "cids": cids.tolist(),
                                   "label_map": lm}}
            except Exception:
                # אם נכשל — artifact בלי פיזור
                art = {"name": "cluster_players", "df": action["df"], "source": src}
        else:
            # שאר הכלים — artifact רגיל (כולל דגלי הבחנה/יעד)
            art = {"name": action["name"], "df": action["df"], "source": src,
                   "disambig": ex.get("disambiguation", False),
                   "target": ex.get("target"),
                   "target_row": ex.get("target_row")}

    # מוסיפים את תשובת העוזר (עם ה-artifact) להיסטוריית התצוגה
    st.session_state.history.append(
        {"role": "assistant", "content": text, "artifact": art})
    # מרעננים כדי להציג את התוצאה
    st.rerun()
