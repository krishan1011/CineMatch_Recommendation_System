# CineMatch

Hybrid movie recommender (content + collaborative filtering + matrix factorization) on MovieLens ml-latest-small.

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
python scripts/download_data.py
pytest
```

## Structure
- `src/` core logic (config, data, metrics, evaluate, models)
- `notebooks/` EDA and experiments
- `app/` Flask web app
- `data/`, `artifacts/`, `reports/` data, trained models, results
- `tests/` pytest suite

## Limitations
Most users rated all their movies within a single session, so many timestamps are tied and the order inside the temporal split is arbitrary for those users.

## Production App
Build the frozen serving artifacts after the processed parquet files and model configs are available:
```bash
python -m src.build_artifacts
```

The compressed engine and lookup artifacts are under 50 MB and are committed with the project so the Flask app can start without fitting models on launch. Raw MovieLens data remains ignored.

Run the app from the project root:
```bash
python app/app.py
```

The app is available at `http://127.0.0.1:5000`. Set `FLASK_DEBUG=1` to enable Flask debug mode.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/search?q=toy` | Search movie titles |
| GET | `/api/similar/<movie_id>?n=10` | Find similar movies |
| GET | `/api/recommend/user/<user_id>?n=10` | Recommend for a MovieLens user |
| POST | `/api/recommend/new?n=12` | Recommend from rated movie IDs and ratings |
| GET | `/api/popular?n=10` | Popular onboarding titles |

Status: Phases 1-10 complete.
