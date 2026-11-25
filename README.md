# Customer Complaints Analytics (NLP demo)

Interactive text analytics on 18,222 consumer complaints about US financial companies
(CFPB public complaint database format, Mar 2015 to Nov 2017).

**Live demo:** TODO add URL after deploying. **Portfolio:** TODO add URL.

## What it shows
- KPI tiles, top-10 companies (click a bar to select), monthly complaint volume
- Keyword-based theme tracker (credit reporting, debt collection, fees, loans, service, fraud)
- Word frequency bars, treemap and wordcloud for any company and month range
- Precomputed LDA topic map with click-to-read complaints
- Bigram comparison between two companies and a GloVe-based bigram similarity map

## Honest notes
- The data is a historical sample bundled with Plotly's app; it is illustrative, not current.
- Themes are keyword rules on bigram flags, not a trained classifier.
- LDA topics were precomputed offline (spaCy + gensim, 5 topics) and overlap heavily; exploratory only.
- The last months of the sample are thin, so the volume trend drops at the end.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:8050
```

## Deploy on Render
Push this folder to GitHub, create a Blueprint from `render.yaml` (free tier sleeps after
15 minutes idle; Starter is $7/month). Set `PORTFOLIO_URL` to show a back link.

## Changes from the original
Rebuilt for Dash 4, dash-bootstrap-components 2 and pandas 3; no spaCy, gensim or matplotlib
at runtime (`legacy/` keeps the offline preprocessing scripts); new KPIs, volume trend and
theme tracker; month-range slider; cached computations; per-request stop-words fix.

## Licence and attribution
MIT. Adapted from Plotly's [dash-nlp](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-nlp)
sample app. Bootstrap 5.3.3 (MIT) is bundled in `assets/`. See `LICENSE`.
