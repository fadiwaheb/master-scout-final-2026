# ⚽ Master Scout

An AI football-player scouting agent. Final project for the AI & ML Innovation
Workshop (course 277302, Academic College of Tel Aviv-Yafo).

**The idea:** combine a *static profile* (EA Sports FC24) with *real match
performance* (Football Events) — an angle standard scouting tools don't offer.
You ask in free text (Hebrew/English); the agent detects intent, extracts
parameters, runs the right model, and returns a short reasoned recommendation.
It recommends only — it stops before action.

## How it works
```
free text → agent (LLM: intent + parameters → JSON) → Python model → NL report
```
The LLM only understands language and writes the report. All analysis is done by
our Python/scikit-learn code (search, Cosine similarity, K-Means, Isolation
Forest / DBSCAN / One-Class SVM).

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your OpenAI key into .env
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Point Streamlit Cloud at this repo, main file `app.py`, and set
`OPENAI_API_KEY` under **Settings → Secrets**.

## Layout
```
app.py                 Streamlit chat app (the live link)
src/agent.py           intent classification + routing (LLM)
src/report.py          natural-language scouting reports (LLM)
src/{search,similarity,clustering,anomaly}.py   the course algorithms
data/processed/final_scouting_table.csv         the central table (18,350 players)
tests/run_demo_tests.py                          demo + documented failure cases
```

> PoC / educational use only. Data has biases and limited coverage (the FC24 vs
> events year gap); scouting decisions affect real people — use with care.
