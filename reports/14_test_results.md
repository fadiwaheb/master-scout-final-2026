# Stage 14 — Test results & demo cases

_Classifier model: `gpt-4.1-nano` · Report model: `gpt-4.1-mini`_

Each case runs the full pipeline: free text -> agent (intent+filters) -> backend function -> natural-language report.


## V1 · Profile search (FC24 only)  — **PASS**

- **Query:** מצא חלוצים מהירים מתחת לגיל 23 עד 30 מיליון יורו, תן 3
- **Classified intent:** `profile_search` (status: `ok`)
- **Expected intent:** profile_search
- **Agent answer:**

  > - Lee Kang In — חלוץ בן 22 עם פוטנציאל גבוה וערך של 28.5 מיליון יורו.
  > - E. Smith Rowe — חלוץ בן 22, מהיר ועם פוטנציאל טוב, שווה 22.5 מיליון יורו.
  > - S. Giménez — חלוץ צעיר בן 22, ערך של 23 מיליון יורו ופוטנציאל גבוה.
  > 
  > ההמלצה מבוססת על פרופיל השחקנים של EA Sports FC24 בלבד.


## V2 · Performance search (real match events)  — **PASS**

- **Query:** מי החלוצים עם הכי הרבה משחקי דאבל (2+ גולים)?
- **Classified intent:** `braces` (status: `ok`)
- **Expected intent:** attacking_players / braces / creative_midfielders
- **Agent answer:**

  > - R. Lewandowski — leads with 35 matches scoring 2+ goals, showing exceptional finishing, based on real match-event performance combined with FC24 profile.
  > - A. Lacazette — 19 matches with 2+ goals, demonstrating strong scoring ability.
  > - A. Griezmann — 18 matches with 2+ goals, consistent double-goal threat.
  > - K. Benzema — 16 matches with 2+ goals, experienced and effective forward.
  > 
  > This is a recommendation only.


## V3 · Similarity (Cosine) — complex, profile + play-style  — **PASS**

- **Query:** who plays like De Bruyne and is worth under 40M?
- **Classified intent:** `similar_players` (status: `ok`)
- **Expected intent:** similar_players
- **Agent answer:**

  > - Brais Méndez — very close similarity (0.9950) with De Bruyne, matching defending, pace, and physic, valued at 30M EUR  
  > - Y. Belhanda — high similarity (0.9945), especially in pace, defending, and physic, very affordable at 3.5M EUR  
  > - B. Bourigeaud — strong similarity (0.9934) in pace, defending, and physic, valued at 25M EUR  
  > - A. Mac Allister — close similarity (0.9917) with emphasis on pace, dribbling, and defending, just under 40M EUR at 39.5M EUR  
  > 
  > These recommendations come from the EA Sports FC24 player profile data. This is a scouting suggestion only.


## V4 · Bargains (anomaly) — high ability vs low value  — **PASS**

- **Query:** find undervalued defenders rated over 80
- **Classified intent:** `bargains` (status: `ok`)
- **Expected intent:** bargains
- **Agent answer:**

  > - G. Chiellini — rated 80 with a market_efficiency_score of 8.28, offering a lot of ability for a low value.
  > - Sergio Ramos — rated 83 with a market_efficiency_score of 5.32, showing good value relative to his quality.
  > - Thiago Silva — rated 84 with a market_efficiency_score of 4.06, still undervalued given his high rating.
  > 
  > These recommendations are based on the EA Sports FC24 player profile and are for consideration only.


## S1 · Out of scope — team question  — **PASS**

- **Query:** כמה כרטיסים צהובים קיבלה ריאל מדריד העונה?
- **Classified intent:** `None` (status: `out_of_scope`)
- **Agent answer:**

  > אני מתמקד בניתוח שחקנים בודדים בלבד ולא בנתוני קבוצות. אשמח לעזור למצוא שחקנים לפי פרופיל, דמיון, סגנון משחק או מציאות.


## S2 · Out of scope — real-world action (stop-before-action)  — **PASS**

- **Query:** תחתים לי את Haaland לקבוצה שלי
- **Classified intent:** `None` (status: `out_of_scope`)
- **Agent answer:**

  > אני מתמקד במציאת שחקנים לפי פרופיל, סגנון, דמיון או מציאות. לא יכול לעזור בהחתמות.


## C1 · Clarify — similarity without a player name  — **PASS**

- **Query:** תמצא לי שחקנים דומים
- **Classified intent:** `similar_players` (status: `clarify`)
- **Agent answer:**

  > כדי להמשיך אני צריך עוד פרט: שם השחקן להשוואה. תוכל להוסיף?


## N1 · Player not in FC24 dataset (retired) — graceful handling  — **PASS**

- **Query:** שחקנים דומים לרונאלדיניו
- **Classified intent:** `similar_players` (status: `not_found`)
- **Agent answer:**

  > I couldn't find 'Ronaldinho' in the FC24 dataset — the player may be retired or spelled differently. Try another name.


## F1 · FAILURE — mixed-intent query the single-intent router can't satisfy  — **DOCUMENTED FAILURE**

- **Query:** מצא חלוצים מהירים שדומים למסי
- **Classified intent:** `similar_players` (status: `ok`)
- **Agent answer:**

  > - P. Dybala — מהיר עם יכולות דומות במקצב, בעיטה וכדרור.
  > - A. Martial — מהיר, עם דגש גם על הגנה וכוח פיזי.
  > - W. Ben Yedder — מהיר, עם יכולות דומות בבעיטה וכוח פיזי.
  > - D. Berardi — מהיר, עם דגש על הגנה ויכולת בעיטה.
  > 
  > ההמלצה מבוססת על פרופיל השחקנים של EA Sports FC24 בלבד.

- **Why it fails (for ch.7):** The user asks for TWO things at once: a profile filter ('fast forwards') AND a similarity target ('like Messi'). Our agent routes each message to exactly ONE intent, so it picks similar_players(Messi) and SILENTLY DROPS the 'fast forwards' constraints — returning players similar to Messi with no guarantee they are fast forwards. Root cause: single-intent, one-shot routing. Fix direction: a multi-step agent loop (Plan->Act) or a post-similarity profile filter — listed as a future improvement, not in the MVP scope.


## F2 · FAILURE — false refusal triggered by the word 'קבוצה' (group/team)  — **DOCUMENTED FAILURE**

- **Query:** תן לי קבוצה של חלוצים טובים
- **Classified intent:** `profile_search` (status: `ok`)
- **Agent answer:**

  > - K. Mbappé — חלוץ צעיר עם פוטנציאל גבוה ויכולת מצוינת, שווה ערך גבוה.
  > - E. Haaland — חלוץ צעיר עם פוטנציאל שווה ל-Mbappé, בעל יכולת התקפית מרשימה.
  > - L. Messi — חלוץ מנוסה עם דירוג גבוה, מתאים למי שמחפש ניסיון ומיומנות.
  > - K. Benzema — חלוץ מנוסה עם דירוג גבוה, ערך סביר יחסית לגילו וליכולת.
  > 
  > ההמלצה מבוססת על פרופיל השחקנים של EA Sports FC24 בלבד.

- **Why it fails (for ch.7):** The user means a GROUP of forwards, but the word 'קבוצה' (which also means 'team') trips the scope boundary and the agent wrongly refuses as a team question. Root cause: keyword-sensitive scope detection over-refuses. Fix direction: rely on the LLM's intent rather than surface keywords, or add 'group of players' as an explicit in-scope example in the system prompt.
