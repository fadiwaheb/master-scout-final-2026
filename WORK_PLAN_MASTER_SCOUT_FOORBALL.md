# Work Plan — Master Scout (Football Player Scouting Agent)

Final project · AI & ML Innovation Workshop · Course 277302 · 2026
Academic College of Tel Aviv-Yafo

This is a stage-by-stage work plan. Each stage is written so you can paste it almost as-is into Claude Code and get a working deliverable. At the end there is a full mapping to the 10 rubric chapters, to maximize the grade.

---

## 0. Locked Decisions (Our Architecture)

- **Agent brain:** OpenAI API (recommended model `gpt-4o-mini` — very cheap, fast, smart enough for intent detection and parameter extraction).
- **Role of the LLM:** (1) intent classification, (2) extracting parameters from free text into JSON, (3) writing the scouting report in natural language. The LLM does **not** compute similarity/clustering/anomalies — our Python algorithms do that.
- **Course algorithms:** TF-IDF + Cosine (similarity), K-Means + DBSCAN (clustering), Isolation Forest + One-Class SVM (anomalies). This is the "dual use" the rubric requires: similarity/clustering → recommendation ; anomalies → flagging.
- **Deployment:** GitHub → Streamlit Community Cloud (free, public live link, runs everything). This is the most important pass/fail condition in the rubric.
- **Language:** Hebrew + English. GPT understands both; the report is returned in the language of the query.
- **Focus:** Player level only. No team analysis, no match prediction, no betting.

### Important rubric note: "Use, don't train"
The rubric says do not train models, but use them and provide the reasoning. Running `KMeans().fit()` or `IsolationForest().fit()` from scikit-learn counts as **using** a ready-made algorithm — that is exactly what is allowed and desired. We are not training a new neural network. In each chapter we explain the reasoning: why this algorithm, which features, which threshold.

---

## Part A — Preliminary Setup (before writing code)

This is the part that is new to us (LLM-based agent + deployment). Do it once.

### A.1 — Development environment and code repository
1. Create a local project folder and a GitHub repo named `master-scout`.
2. Create a virtual environment (`python -m venv venv`) and install: `pandas, numpy, scikit-learn, streamlit, openai, python-dotenv, matplotlib, seaborn`.
3. Add `requirements.txt` (Streamlit Cloud reads it automatically) and `.gitignore` (include `.env`, `venv/`, heavy data files).

### A.2 — OpenAI key (the brain layer)
1. Open an account at platform.openai.com, create an API key, and load a small amount of credit (gpt-4o-mini costs a few cents for the whole project).
2. **Never write the key in the code.** Locally — in a `.env` file. On Streamlit Cloud — under *Settings → Secrets*.
3. Test "Hello World": a short script that sends a prompt and returns a response, to confirm the connection works.

### A.3 — Downloading the data (Kaggle)
- Source 1: `male_players.csv` from EA Sports FC 24 Complete Player Dataset.
- Source 2: `events.csv` (optionally `ginf.csv`, `dictionary.txt`) from Football Events.
- Download manually from Kaggle and save in `data/raw/`. Do not push the heavy files to GitHub (keep them in `.gitignore`); attach a separate ZIP in the LMS submission as the rubric requires.

### A.4 — Streamlit skeleton + initial deploy (verify the link works early)
1. Create a minimal `app.py` with a single text box that returns "hello".
2. Push to GitHub → on Streamlit Community Cloud connect the repo → get a public link.
3. Verify an external person (a friend's phone) can open the link. **Do this early** to avoid panic at the end.

---

## Part B — Development Stages (feed into Claude Code, stage by stage)

Each stage: objective · input · output · functions · quality checks · what to ask Claude Code · stage completion.

### Stage 1 — Understand file structure and define columns
- **Objective:** Map which columns from `male_players.csv` and `events.csv` are relevant, and document the meaning of the codes in events.
- **Input:** `data/raw/male_players.csv`, `data/raw/events.csv`, `dictionary.txt`.
- **Output:** A doc/notebook `01_schema.md` with the chosen column list + a code mapping table (e.g., `event_type=1 → Attempt`).
- **Functions:** None yet — exploration only.
- **Quality checks:** Count rows and columns in each file; confirm there are >10 columns and tens of thousands of records (the Big Data requirement in the rubric).
- **Ask Claude Code:** "Load both files, print shape, dtypes, and first 5 rows, and build me a code mapping table from dictionary.txt."
- **Completion:** A final column list + a documented code table.

### Stage 2 — Load and initial quality check (EDA)
- **Objective:** Descriptive statistics + detect missing/outliers + charts — basis for rubric chapter 4.
- **Input:** Raw files.
- **Output:** `02_eda.ipynb`, several charts saved to `reports/figures/` (age/value/overall distributions; event counts by type).
- **Functions:** `load_players_data(path)`, `load_events_data(path)`.
- **Quality checks:** Percentage of missing values per column; mean/median/std for key numeric columns.
- **Ask Claude Code:** "Produce descriptive statistics, missing-value percentages, and 4 basic charts, and save them as PNG files."
- **Completion:** An EDA report + saved charts that can go into the submission file.

### Stage 3 — Clean players → `clean_players`
- **Objective:** A clean player-level table + computed score columns.
- **Input:** `male_players.csv`.
- **Output:** `data/processed/clean_players.csv` (row = player).
- **Functions:** `clean_player_name(name)`, `clean_players_data(df)`, `calculate_ability_score`, `calculate_market_efficiency_score`, and the other scores from the spec.
- **Quality checks:** No duplicate `player_id`; no NaN in key columns (overall, value_eur); `position_group` computed correctly.
- **Ask Claude Code:** "Clean names, select relevant columns, create position_group and computed score columns, save clean_players.csv."
- **Completion:** The file exists, reloads without errors, reasonable row count.

### Stage 4 — Clean events + map codes → `clean_events`
- **Objective:** A clean events table with readable text columns and binary columns.
- **Input:** `events.csv` + the code table.
- **Output:** `data/processed/clean_events.csv` (row = event).
- **Functions:** `clean_events_data(df)`, `map_event_codes(df)` (creates `event_type_name`, `is_goal`, `is_key_pass`, `is_box_shot`, `is_left_foot`, etc.).
- **Quality checks:** Sum of `is_goal` is reasonable; all codes were translated (no empty `*_name`); no rows with empty `player` in critical places.
- **Ask Claude Code:** "Convert all event codes to text names, create binary columns per the spec, clean and save clean_events.csv."
- **Completion:** A clean events file with valid binary columns.

### Stage 5 — `player_match_stats` (player × match)
- **Objective:** Aggregation at the player-in-match level (basis for braces, key passes, etc.).
- **Input:** `clean_events.csv`.
- **Output:** `data/processed/player_match_stats.csv`.
- **Functions:** `build_player_match_stats(clean_events_df)`.
- **Quality checks:** Goals per match ≤ number of events; correct grouping by `clean_player_name + id_odsp`.
- **Ask Claude Code:** "Aggregate events to player-in-match level: shots, goals, key_passes, cards, box_shots, foot shots, etc."
- **Completion:** A valid table that reconciles against clean_events.

### Stage 6 — `player_event_stats` (aggregated player)
- **Objective:** One row per player with all aggregated and ratio performance metrics.
- **Input:** `player_match_stats.csv`.
- **Output:** `data/processed/player_event_stats.csv`.
- **Functions:** `build_player_event_stats(df)`, `calculate_attacking_involvement_score`, `calculate_creative_score`, `calculate_discipline_score`, `calculate_foot_balance_score`, plus per_match and rate columns, and `matches_with_2_plus_goals`.
- **Quality checks:** `goals_per_match = total_goals/matches`; rates between 0 and 1; no division by zero.
- **Ask Claude Code:** "Summarize to player level: totals, per_match, rates, scores, and matches_with_2_plus_goals."
- **Completion:** A complete aggregated player table.

### Stage 7 — `final_scouting_table` (the central table)
- **Objective:** Merge FC 24 profile + event metrics, with source marking — handles the year/coverage gap.
- **Input:** `clean_players.csv` + `player_event_stats.csv`.
- **Output:** `data/processed/final_scouting_table.csv` (row = player).
- **Functions:** `build_final_scouting_table(...)` that adds `has_event_data` and `data_source_note`.
- **Quality checks:** Players without an event match are kept with `has_event_data=False`; no duplicate players.
- **Ask Claude Code:** "Merge the two tables with a left join on clean name, add has_event_data and data_source_note, save final_scouting_table.csv."
- **Completion:** One central table that the entire agent runs on.

### Stage 8 — Search and filter functions (intents 1, 2, 5, 6, 7, 8)
- **Objective:** Parameter-based search over the central table.
- **Input:** `final_scouting_table.csv`.
- **Output:** module `search.py`.
- **Functions:** `search_players_by_profile(filters)`, `search_attacking_players`, `search_creative_midfielders`, `search_disciplined_defenders`, `search_two_footed_players`, `find_players_with_min_braces(min_braces)`.
- **Quality checks:** Every threshold is a **parameter** (max_age, min_pace, max_value_eur, min_braces, top_n...) that is easy to change; results filtered correctly.
- **Ask Claude Code:** "Build modular search functions, every threshold as a parameter with a default, returning a sorted DataFrame."
- **Completion:** All six searches work on examples.

### Stage 9 — Cosine Similarity for similar players (intent 3) — **Use #1**
- **Objective:** Find similar players + the comparison for the rubric.
- **Input:** The central table.
- **Output:** `similarity.py`.
- **Functions:** `find_similar_players(player_name, filters, top_n)` with feature normalization and Cosine.
- **Model comparison (critical for chapter 5+6, 20 pts):** Run once **with** position_group as a category (filter to the same position) and once **without** — exactly like exercise 2 in the course. Document the difference in results.
- **Quality checks:** Similarity score between 0 and 1; a player is not returned as similar to himself; a short text explanation of why they are similar.
- **Ask Claude Code:** "Normalize features, compute Cosine against a target player, return top_n + score, and add a comparison run with/without category."
- **Completion:** A sensible similar-players list + comparative output for documentation.

### Stage 10 — K-Means for play-style clustering (intent 4) — **Use #1 (clustering)**
- **Objective:** Group players into styles (fast strikers, technical midfielders, physical center-backs...).
- **Input:** The central table (selected features).
- **Output:** `clustering.py` + a `cluster_id` column in the table.
- **Functions:** `run_player_kmeans(n_clusters)`, `get_players_from_cluster(cluster_id, filters)`, `describe_cluster(cluster_id)`.
- **Quality checks:** Choose k using the Elbow method — document the chart; clusters are logically interpretable.
- **Ask Claude Code:** "Run K-Means on normalized features, find k via the Elbow method, label each player, and write a verbal description for each cluster."
- **Completion:** Explainable clusters + an Elbow chart for documentation.

### Stage 11 — Anomaly detection: Isolation Forest + DBSCAN + One-Class SVM (intents 4, 9) — **Use #2**
- **Objective:** Detect bargain players and profile-vs-performance anomalies.
- **Input:** The central table.
- **Output:** `anomaly.py`.
- **Functions:** `detect_bargain_players(filters)`, `detect_profile_performance_anomalies(filters)`, `run_isolation_forest(features)`, `run_dbscan(features)`.
- **Model comparison (again contributes to chapter 5+6):** Compare Isolation Forest vs DBSCAN on the same features, and document which caught which outliers. Optionally add One-Class SVM as a third for bonus.
- **Quality checks:** `anomaly_contamination` as a parameter; returned outliers are indeed far from the cluster; "bargain" logic = high ability vs low value.
- **Ask Claude Code:** "Implement Isolation Forest and DBSCAN to detect outliers and bargains, contamination as a parameter, and a comparison between them."
- **Completion:** Bargain/outlier lists + a model-comparison output.

### Stage 12 — The Agent layer (GPT) — the new part
- **Objective:** Turn free text (Hebrew/English) into action: intent detection → parameter extraction → routing to a function.
- **Input:** A free-text user message.
- **Output:** `agent.py`.
- **Functions:** `classify_user_intent(query)`, `extract_filters_from_query(query)`, `validate_scope(query)`, `ask_clarifying_question(missing_fields)`, `route_query_to_function(intent, filters)`.
- **Prompt design (the heart of the agent):**
  - **System prompt** defines: identity ("Master Scout"), the domain (player analysis only), the 10 intents, and the rule to refuse out-of-domain questions.
  - Ask GPT to return **JSON only** in a fixed schema: `{"intent": "...", "filters": {...}, "missing": [...], "in_scope": true/false}`.
  - **Scope boundary:** if `in_scope=false` → the agent politely refuses (refusal wording from the spec).
  - **Source order:** our tables first → if not found, state that external search is needed (do not search automatically in the first stage).
  - **Stop before action:** the agent never "signs" a player — it ends with a recommendation only.
- **Quality checks:** The 10 demo queries from the spec are classified to the correct intent; the "ticket to a Barcelona match" question is refused.
- **Ask Claude Code:** "Build an agent layer that uses OpenAI gpt-4o-mini, returns JSON with intent+filters+missing+in_scope, and routes to the existing functions. The key is loaded from secrets, not in the code."
- **Completion:** Free text turns into the correct function call, including out-of-scope refusal and a clarifying question when info is missing.

### Stage 13 — Natural-language scouting reports (NLG)
- **Objective:** Turn table results into a readable report in Hebrew/English with reasoning.
- **Input:** Function output + the original user query.
- **Output:** `report.py`.
- **Functions:** `generate_scouting_report(results, user_query)`.
- **Quality checks:** The report is in the language of the query; states whether the info is from FC 24 only or also from Football Events (`has_event_data`); explains the similarity score / reason for the anomaly.
- **Ask Claude Code:** "Give GPT the results as context and ask for a short, reasoned report in the language of the query, stating the data source."
- **Completion:** A clear, reasoned report for each of the 10 intents.

### Stage 14 — Tests and demo cases (rubric chapter 7)
- **Objective:** ≥3 test cases, including a **documented failure case** — an explicit, scored requirement.
- **Input:** The full agent.
- **Output:** `tests/` + a screenshots folder.
- **Content:** (1) a valid case (profile search), (2) a complex valid case (profile+performance), (3) an out-of-scope case that is refused, (4) a **failure case** — an ambiguous/poorly worded query that the agent misidentifies, with an explanation of why it failed.
- **Ask Claude Code:** "Write a test script that runs the 4 scenarios and saves outputs; also craft a deliberate failure scenario."
- **Completion:** Screenshots for each case including the failure — ready to paste into chapter 7.

### Stage 15 — Live deployment (the most important pass/fail condition)
- **Objective:** A public link the lecturer opens and runs.
- **Input:** All the code + `final_scouting_table.csv`.
- **Output:** A live Streamlit app + a link.
- **Content:** `app.py` with a chat box; load the processed table; connect to `agent.py`; OpenAI key in *Secrets*; push to GitHub; connect to Streamlit Cloud.
- **Quality checks:** An external person opens the link and runs a full end-to-end query.
- **Ask Claude Code:** "Build app.py in Streamlit with a simple chat interface that connects all the layers, and loads the OpenAI key from st.secrets."
- **Completion:** The link works from a foreign device — the project is "live".

---

## Part C — Mapping to the Rubric Chapters (to get full credit)

The submission file (up to 10 pages) must be organized **exactly** in this numbering. The lecturer grades chapter by chapter.

1. **Domain background (10):** Football scouting; 2 sources (FC 24 + Football Events), record counts, the year gap as a PoC; problem statement and goals.
2. **Innovation and originality (10):** Combining a static profile (FC 24) with actual performance from match events — an angle that does not exist in standard scouting tools.
3. **AI technical depth (10):** Concrete explanation — TF-IDF+Cosine for similar players, K-Means/DBSCAN for clustering, Isolation Forest for anomalies; the dual use; GPT for the language layer.
4. **Big Data (10):** Merging 2 sources, cleaning, the 7 tables, descriptive statistics + a **data flow diagram** (raw → clean → stats → final).
5+6. **Implementation — model choice and application (20, the heavy one):** Selection reasoning; **comparison of 2 models** (Cosine with/without category; Isolation Forest vs DBSCAN); **end-to-end pseudocode**; result examples.
7. **Testing and validation (10):** The 4 cases from stage 14, including the failure case, with screenshots.
8. **Ethical considerations (10):** Specific — data bias (FC 24 is subjective; partial player coverage), the risk of life-changing decisions about a player based on numbers, a warning that this is a PoC.
9. **Conclusions and application (10):** Two sharp conclusions from the actual results + assumptions and limitations (year gap, name matching).
10. **Impact assessment (10):** Extension to other leagues/positions/clubs; advantages and challenges in the scouting industry.

---

## Part D — Scoring Strategy and Bonuses

- **The two sections that earn the most:** Chapter 5+6 (20 pts — invest in the model comparison and the broad pseudocode) and chapter 7 (the documented failure case).
- **MVP first:** Finish stages 1–15 at a basic level with a live link **before** adding bonuses.
- **Recommended bonuses (if time allows):** Adding One-Class SVM as a third anomaly model for comparison; an Elbow chart and high-quality visualizations; a full Agent loop (Plan→Act→Observe) instead of one-shot routing; RAG over player descriptions (Embeddings + Vector DB) — session 3.

---

## Part E — Pre-Submission Checklist

- [ ] A live Streamlit link that opens from a foreign device and runs a full query.
- [ ] 3 deliverables to LMS: full rubric · work file by chapters 1–10 · code (GitHub) + data (ZIP).
- [ ] The dual use is clear in the file: similarity/clustering (Cosine, K-Means) + anomalies (Isolation Forest).
- [ ] A documented comparison of 2 models + end-to-end pseudocode.
- [ ] A documented failure case with a screenshot.
- [ ] Scope boundary works (refuses an out-of-domain question) + stops before action.
- [ ] The OpenAI key is in Secrets only, not in the code.
- [ ] Presentation prep: ~30 seconds on the project and the data, then a live demo (no slides).
