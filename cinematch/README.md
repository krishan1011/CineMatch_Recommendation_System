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

Status: Phase 1 complete (skeleton + config).
