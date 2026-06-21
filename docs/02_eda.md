# 02 — Exploratory Data Analysis (Stage 2)

Master Scout · Output of Stage 2. Basis for rubric chapter 4 (Big Data).
Reproduce with: `conda activate masterscout && python scripts/run_eda.py`

---

## Dataset sizes

| Dataset | Rows | Cols | Notes |
|---|---|---|---|
| FC24 players (`fifa_version==24`) | **18,350** | 52 | one row per player |
| Football Events | **941,009** | 18 | one row per match event; 9,074 matches, 6,118 players |

Big Data requirement (>10 columns, tens/hundreds of thousands of rows): ✅ comfortably met.

---

## Descriptive statistics — FC24 players

| Metric | mean | median | std | min | max |
|---|---|---|---|---|---|
| age | 25.27 | 25 | 4.76 | 16 | 43 |
| overall | 65.82 | 66 | 6.82 | 47 | 91 |
| potential | 71.09 | 71 | 6.22 | 48 | 94 |
| value_eur | 2,837,585 | 1,000,000 | 7,562,794 | 10,000 | 185,000,000 |
| wage_eur | 8,723 | 3,000 | 18,707 | 500 | 350,000 |
| pace | 68.37 | 69 | 10.77 | 27 | 97 |
| shooting | 52.58 | 55 | 13.90 | 19 | 93 |
| passing | 57.49 | 58 | 9.90 | 25 | 94 |
| dribbling | 62.84 | 64 | 9.45 | 28 | 94 |
| defending | 52.09 | 57 | 16.03 | 15 | 89 |
| physic | 64.90 | 66 | 9.92 | 32 | 89 |

**Reading it:**
- `value_eur` is **heavily right-skewed** (mean 2.8M vs median 1.0M, max 185M). This is why we plot it on a log scale and why a "bargain" detector (Stage 11) makes sense — most players are cheap, a few are extreme.
- `overall` is roughly normal around 66 — a healthy spread for similarity/clustering.
- `defending` has the widest spread (std 16) — it cleanly separates defenders from attackers, useful for the play-style clusters.

---

## Missing values

### FC24 players
| Column | Missing | % |
|---|---|---|
| pace, shooting, passing, dribbling, defending, physic | 2,045 each | 11.14 |
| release_clause_eur | 1,280 | 6.98 |
| value_eur | 100 | 0.54 |
| wage_eur / club_name / league_name | 87 | 0.47 |

**Key finding — the 11.14% is NOT a data-quality problem.** It is exactly the **2,045 goalkeepers**: the 6 outfield face stats (pace…physic) are blank for every GK and for no outfield player (verified: 2,045 GKs, all 6 stats NaN; 0 outfield players missing pace). Since the agent focuses on **outfield play styles**, GKs will be filtered or handled separately in cleaning (Stage 3).
- `release_clause_eur` missing (7%): players without a release clause in their contract — fill or leave as "no clause".
- `value/wage/club` missing (~0.5%): free agents / unattached players.

### Football Events
| Column | Missing | % | Reason |
|---|---|---|---|
| event_type2 | 726,716 | 77.2 | only set for key-pass / through-ball / sending-off / own-goal events |
| shot_place, shot_outcome, situation, bodypart | ~712,000 | ~75.7 | **only apply to shots** (`event_type==1`); blank for fouls, corners, etc. |
| player2 | 649,699 | 69.0 | only for events with a second player (assist, fouled player) |
| location | 473,942 | 50.4 | only recorded for shots and some set pieces |
| player | 61,000 | 6.5 | team-level events with no individual (e.g. some corners) |

**Key finding — the high missingness is structural, not corruption.** These columns are conditional on event type. When we aggregate to player level (Stages 5–6) we count events, so blanks become legitimate zeros — no imputation needed.

---

## Event composition
- Total goals (`is_goal==1`): **24,446**
- Most common events: Free kick won, Foul, Attempt (shots), Corner.
- Cards and penalties are rare (good — they make strong "discipline" signals).

---

## Charts (in `reports/figures/`)
| File | Shows |
|---|---|
| `01_age_distribution.png` | Age — peaks ~21–26, long tail to 43 |
| `02_overall_distribution.png` | Overall — bell around 66 |
| `03_value_distribution.png` | Market value (log) — extreme right skew |
| `04_event_counts.png` | Events by type — fouls/free-kicks/shots dominate |
| `05_overall_vs_value.png` | Overall vs value — value explodes above ~80 overall (the "premium" bargains exploit) |
| `06_shots_by_bodypart.png` | Shots by foot/head — basis for the two-footed feature |

---

## Implications for later stages
1. **Filter goalkeepers** (or keep, but never feed NaN face-stats into similarity/clustering). — Stage 3
2. **No imputation on event columns** — aggregate counts turn blanks into zeros. — Stages 5–6
3. **Log-transform / robust scaling for value** before anomaly detection. — Stage 11
4. **`release_clause_eur`** — treat missing as "no clause", not zero value. — Stage 3

## Stage 2 completion checklist
- [x] Descriptive statistics (mean/median/std/min/max) for key numerics
- [x] Missing-value percentages per column, with root-cause explained
- [x] 6 charts saved as PNG to `reports/figures/`
- [x] Outliers/skew identified (value_eur) — feeds the bargain detector

**Next:** Stage 3 — clean players → `data/processed/clean_players.csv`.
