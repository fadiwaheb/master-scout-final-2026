"""
external.py — REAL external data source: the EA Sports FC official ratings API.

The ratings site https://www.ea.com/games/ea-sports-fc/ratings is backed by a
public JSON API (drop-api.ea.com). We query it directly — no fragile HTML
scraping, no browser — to fetch live, accurate attributes (and a player photo)
for players that are NOT in our local datasets.

Source priority in the agent: our FC24 data -> our events data -> THIS (EA FC
official ratings) -> the model's own football knowledge (last resort).
"""
from __future__ import annotations

import datetime as _dt

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

EA_API = "https://drop-api.ea.com/rating/ea-sports-fc"
SOURCE_NAME = "EA Sports FC — דירוגים רשמיים"
SOURCE_URL = "https://www.ea.com/games/ea-sports-fc/ratings"

_POS = {"attack": "Forward", "midfield": "Midfielder",
        "defense": "Defender", "goalkeeper": "GK"}
_FOOT = {1: "Right", 2: "Left"}


def _age(birthdate: str):
    try:
        d = _dt.datetime.strptime(birthdate.split()[0], "%m/%d/%Y").date()
        t = _dt.date.today()
        return t.year - d.year - ((t.month, t.day) < (d.month, d.day))
    except Exception:
        return None


def ea_fc_lookup(name: str, timeout: int = 8) -> dict | None:
    """Look a player up in EA Sports FC's official ratings API.
    Returns a dict in our card schema (with avatar_url + source), or None."""
    if requests is None or not name:
        return None
    try:
        r = requests.get(EA_API, params={"locale": "en", "limit": 1, "search": name},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        items = r.json().get("items", [])
    except Exception:
        return None
    if not items:
        return None

    it = items[0]
    s = it.get("stats", {}) or {}

    def stat(k):
        v = s.get(k)
        return v.get("value") if isinstance(v, dict) else None

    full = f"{it.get('firstName', '')} {it.get('lastName', '')}".strip()
    pos_type = ((it.get("position") or {}).get("positionType") or {}).get("id")
    return {
        "short_name": it.get("commonName") or full or name,
        "long_name": full or it.get("commonName"),
        "position_group": _POS.get(pos_type),
        "age": _age(it.get("birthdate", "")),
        "overall": it.get("overallRating"),
        "potential": it.get("overallRating"),       # API has no potential
        "value_eur": None,                           # API has no market value
        "preferred_foot": _FOOT.get(it.get("preferredFoot")),
        "nationality_name": (it.get("nationality") or {}).get("label"),
        "club_name": (it.get("team") or {}).get("label"),
        "pace": stat("pac"), "shooting": stat("sho"), "passing": stat("pas"),
        "dribbling": stat("dri"), "defending": stat("def"), "physic": stat("phy"),
        "avatar_url": it.get("avatarUrl"),
        "nationality_flag_url": (it.get("nationality") or {}).get("imageUrl"),
        "club_badge_url": (it.get("team") or {}).get("imageUrl"),
        "has_event_data": False,
        "_source": "ea", "_source_url": SOURCE_URL,
    }


def _strip_initial(name: str) -> str:
    """'L. Messi' -> 'Messi' — EA search matches the surname, not the 'X.' form."""
    import re
    return re.sub(r"^[A-Z]\.\s*", "", str(name)).strip()


def ea_media(name: str) -> dict:
    """Lightweight: just the photo + flag + badge for a player (to enrich the cards
    of players that ARE in our data). Tries the name and its surname form."""
    for q in (name, _strip_initial(name)):
        d = ea_fc_lookup(q)
        if d and d.get("avatar_url"):
            return {"avatar_url": d.get("avatar_url"),
                    "flag_url": d.get("nationality_flag_url"),
                    "badge_url": d.get("club_badge_url")}
    return {}
