from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BI_DIR = PROJECT_ROOT / "reports" / "bi"
REQUIRED_ARTIFACTS = [
    PROJECT_ROOT / "artifacts" / "engine.joblib",
    PROJECT_ROOT / "artifacts" / "seen_by_user.joblib",
    PROJECT_ROOT / "artifacts" / "id_maps.joblib",
]
BI_COLUMNS = {
    "bi_ratings_summary.csv": ["summary_type", "key", "count", "mean_rating"],
    "bi_movies.csv": ["movie_id", "title", "year", "genre", "rating_count", "mean_rating", "popularity_decile"],
    "bi_users.csv": ["user_id", "n_ratings", "mean_rating", "activity_bucket", "first_rating_date", "last_rating_date", "activity_span_days"],
    "bi_sorted_counts.csv": ["rank", "movie_id", "rating_count", "cumulative_share"],
    "bi_kpis.csv": ["n_users", "n_movies", "n_ratings", "sparsity"],
    "results_test.csv": ["model", "split", "metric", "value", "params"],
    "results_val.csv": ["model", "split", "metric", "value", "params"],
    "bi_model_labels.csv": ["model", "label", "is_hybrid"],
    "bi_error_by_bucket.csv": ["model", "bucket_type", "bucket", "bucket_order", "rmse", "ndcg"],
    "bi_cold_start.csv": ["model", "n_ratings", "ndcg_at_10"],
    "bi_recs_sample.csv": ["user_id", "model", "rank", "movie_id", "title", "genres"],
    "bi_popularity_bias.csv": ["model", "popularity_decile", "recommendation_share", "test_relevant_share", "top_decile_recommendation_share", "top_decile_test_relevant_share"],
    "bi_diversity.csv": ["model", "mean_intra_list_diversity", "users_evaluated"],
}

if not all(path.exists() for path in REQUIRED_ARTIFACTS):
    pytest.skip("production artifacts are not built", allow_module_level=True)
if not all((BI_DIR / name).exists() for name in BI_COLUMNS):
    pytest.skip("BI exports are not generated", allow_module_level=True)


@pytest.mark.parametrize("filename,expected_columns", BI_COLUMNS.items())
def test_bi_export_exists_is_nonempty_and_has_expected_columns(filename, expected_columns):
    path = BI_DIR / filename

    assert path.is_file()
    assert path.stat().st_size > 0
    frame = pd.read_csv(path)
    assert not frame.empty
    assert frame.columns.tolist() == expected_columns
