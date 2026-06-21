# 01 — Schema & Column Definitions (Stage 1)

Master Scout · Football Player Scouting Agent
Output of Stage 1: file structure mapping + event code dictionary.

---

## Data sources overview

| File | Source | Rows | Key entity | Notes |
|---|---|---|---|---|
| `data/raw/male_players.csv` | EA Sports FC 24 — Complete Player Dataset (Kaggle) | **180,021** | player × fifa_version | 109 columns. Contains **10 FIFA versions (15–24)**, **53,111 unique players**. FC24 alone = 18,350 players. |
| `data/raw/events.csv` | Football Events (Kaggle) | **941,009** | match event | 22 columns. **9,074 matches**, **6,118 players**, **24,446 goals**. European leagues ~2011/12–2016/17. |

**Big Data requirement (rubric ch. 4):** ✅ both files have >10 columns and tens/hundreds of thousands of records.

**The "year gap" (PoC angle):** FC24 profile is from 2023; events are from ~2012–2017. Player matching is by cleaned name, so only players present in both eras get `has_event_data=True`. This gap is a documented limitation and the core proof-of-concept point.

---

## Part 1 — `male_players.csv` (FC 24 static profile)

**Decision:** filter to `fifa_version == 24` to get one current profile per player (row = player).

### Chosen columns

**Identity / meta**
- `player_id`, `short_name`, `long_name`, `player_positions`, `age`, `dob`, `nationality_name`, `club_name`, `league_name`, `preferred_foot`

**Market / value**
- `value_eur`, `wage_eur`, `release_clause_eur`, `overall`, `potential`, `international_reputation`

**Physical**
- `height_cm`, `weight_kg`, `weak_foot`, `skill_moves`, `work_rate`, `body_type`

**Core attribute ratings (the 6 FC face stats)**
- `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`

**Detailed attributes (selected, for scoring)**
- `attacking_finishing`, `attacking_short_passing`, `attacking_crossing`, `attacking_heading_accuracy`
- `skill_dribbling`, `skill_ball_control`, `skill_long_passing`, `skill_fk_accuracy`
- `movement_acceleration`, `movement_sprint_speed`, `movement_agility`, `movement_reactions`
- `power_shot_power`, `power_stamina`, `power_strength`, `power_long_shots`
- `mentality_vision`, `mentality_positioning`, `mentality_interceptions`, `mentality_composure`
- `defending_standing_tackle`, `defending_sliding_tackle`, `defending_marking_awareness`

**Dropped:** the 26 positional rating columns (`ls, st, rs … gk`), all `goalkeeping_*` (we focus on outfield play styles), `player_url`, `fifa_update`, `update_as_of`, `*_team_id`, jersey numbers, loan/contract dates, `player_tags`, `real_face`.

### `position_group` (computed in Stage 3)
Derived from the first token of `player_positions`:
| Group | FC positions |
|---|---|
| GK | GK |
| Defender | CB, RB, LB, RWB, LWB |
| Midfielder | CDM, CM, CAM, RM, LM |
| Forward | RW, LW, CF, ST |

---

## Part 2 — `events.csv` (Football Events — actual match performance)

### Chosen columns
`id_odsp` (match id), `time`, `text`, `event_type`, `event_type2`, `side`, `event_team`, `opponent`, `player`, `player2`, `shot_place`, `shot_outcome`, `is_goal`, `location`, `bodypart`, `assist_method`, `situation`, `fast_break`.

`player` is already lowercased in the raw file (e.g. `mladen petric`) — convenient for name matching.

---

## Part 3 — Event code dictionary (the mapping table)

Verified against the actual unique values present in the file.

### `event_type` (main event)
| Code | Meaning |
|---|---|
| 1 | Attempt (shot) |
| 2 | Corner |
| 3 | Foul |
| 4 | Yellow card |
| 5 | Second yellow card |
| 6 | (Straight) Red card |
| 7 | Substitution |
| 8 | Free kick won |
| 9 | Offside |
| 10 | Hand ball |
| 11 | Penalty conceded |

### `event_type2` (secondary event)
| Code | Meaning |
|---|---|
| 12 | Key Pass |
| 13 | Failed through ball |
| 14 | Sending off |
| 15 | Own goal |

### `side`
| Code | Meaning |
|---|---|
| 1 | Home |
| 2 | Away |

### `shot_place`
| Code | Meaning | | Code | Meaning |
|---|---|---|---|---|
| 1 | Bit too high | | 8 | Misses to the left |
| 2 | Blocked | | 9 | Misses to the right |
| 3 | Bottom left corner | | 10 | Too high |
| 4 | Bottom right corner | | 11 | Top centre of the goal |
| 5 | Centre of the goal | | 12 | Top left corner |
| 6 | High and wide | | 13 | Top right corner |
| 7 | Hits the bar | | | |

### `shot_outcome`
| Code | Meaning |
|---|---|
| 1 | On target |
| 2 | Off target |
| 3 | Blocked |
| 4 | Hit the bar |

### `location` (shot location)
| Code | Meaning | | Code | Meaning |
|---|---|---|---|---|
| 1 | Attacking half | | 11 | Right side of the box |
| 2 | Defensive half | | 12 | Right side of the six yard box |
| 3 | Centre of the box | | 13 | Very close range |
| 4 | Left wing | | 14 | Penalty spot |
| 5 | Right wing | | 15 | Outside the box |
| 6 | Difficult angle and long range | | 16 | Long range |
| 7 | Difficult angle on the left | | 17 | More than 35 yards |
| 8 | Difficult angle on the right | | 18 | More than 40 yards |
| 9 | Left side of the box | | 19 | Not recorded |
| 10 | Left side of the six yard box | | | |

### `bodypart`
| Code | Meaning |
|---|---|
| 1 | Right foot |
| 2 | Left foot |
| 3 | Head |

### `assist_method`
| Code | Meaning |
|---|---|
| 0 | None |
| 1 | Pass |
| 2 | Cross |
| 3 | Headed pass |
| 4 | Through ball |

### `situation`
| Code | Meaning |
|---|---|
| 1 | Open play |
| 2 | Set piece |
| 3 | Corner |
| 4 | Free kick |

### `fast_break`
| Code | Meaning |
|---|---|
| 0 | No |
| 1 | Yes |

---

## Binary columns to derive in Stage 4 (`map_event_codes`)
| New column | Rule |
|---|---|
| `is_shot` | `event_type == 1` |
| `is_goal` | already present (use as-is) |
| `is_key_pass` | `event_type2 == 12` |
| `is_box_shot` | `location` in {3, 9, 10, 11, 12, 13, 14} (inside the box) |
| `is_left_foot` | `bodypart == 2` |
| `is_right_foot` | `bodypart == 1` |
| `is_header` | `bodypart == 3` |
| `is_on_target` | `shot_outcome == 1` |
| `is_yellow` | `event_type == 4` |
| `is_red` | `event_type` in {5, 6} |
| `is_foul` | `event_type == 3` |
| `is_through_ball_assist` | `assist_method == 4` |

---

## Stage 1 completion checklist
- [x] Both files load; shapes, dtypes, head inspected.
- [x] Final column lists chosen for each file.
- [x] Event code mapping table built and **verified against actual values**.
- [x] Big Data requirement confirmed (>10 cols, tens of thousands of rows).
- [x] Year-gap / PoC limitation documented.

**Next:** Stage 2 — EDA (descriptive stats, missing values, 4 charts).
