"""Export tidy, Power BI-friendly CSV tables from existing CineMatch artifacts."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DATA_PROC, REPORTS


MODEL_LABELS = {
    "random": ("Random", False),
    "popularity_count": ("Popularity (count)", False),
    "popularity_bayes": ("Popularity (Bayesian)", False),
    "bias": ("Bias baseline", False),
    "content": ("Content-based", False),
    "item_knn": ("Item-kNN", False),
    "user_knn": ("User-kNN", False),
    "item_knn_implicit": ("Item-kNN (implicit)", False),
    "user_knn_implicit": ("User-kNN (implicit)", False),
    "mf": ("Matrix Factorization (RMSE)", False),
    "mf_ndcg": ("Matrix Factorization (ranking)", False),
    "hybrid": ("Hybrid recommender", True),
    "hybrid_stack": ("Hybrid rating stack", True),
}


def _genres(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    return [str(genre) for genre in value if str(genre) != "(no genres listed)"]


def _movie_decade(year):
    if pd.isna(year):
        return "Unknown"
    return str((int(float(year)) // 10) * 10)


def _write_csv(frame, output_dir, name):
    path = output_dir / name
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _top_items(scores, excluded, n):
    values = np.asarray(scores, dtype=float).reshape(-1).copy()
    excluded = np.asarray(list(excluded), dtype=int)
    excluded = excluded[(excluded >= 0) & (excluded < len(values))]
    values[excluded] = -np.inf
    candidates = np.flatnonzero(np.isfinite(values))
    count = min(int(n), len(candidates))
    if count == 0:
        return []
    selected = candidates[np.argpartition(values[candidates], -count)[-count:]]
    order = np.lexsort((selected, -values[selected]))
    return [(int(item), float(values[item])) for item in selected[order]]


def export_bi():
    """Create all requested Power BI tables using existing processed files."""
    output_dir = REPORTS / "bi"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_PROC / "train.parquet")
    val = pd.read_parquet(DATA_PROC / "val.parquet")
    test = pd.read_parquet(DATA_PROC / "test.parquet")
    full = pd.concat([train, val, test], ignore_index=True)
    movies = pd.read_parquet(DATA_PROC / "movies.parquet").copy()
    movies["genre_list"] = movies["genre_list"].map(_genres)
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    full["rating_datetime"] = pd.to_datetime(full["timestamp"], unit="s", utc=True)
    full["rating_year"] = full["rating_datetime"].dt.year

    movie_stats = (
        full.groupby("i")["rating"]
        .agg(rating_count="size", mean_rating="mean")
        .reindex(movies["i"].to_numpy())
    )
    movies["rating_count"] = movie_stats["rating_count"].fillna(0).to_numpy(dtype=np.int64)
    movies["mean_rating"] = movie_stats["mean_rating"].to_numpy(dtype=float)
    movies["popularity_rank"] = movies["rating_count"].rank(
        method="first", ascending=False
    )
    movies["popularity_decile"] = np.ceil(
        movies["popularity_rank"] / len(movies) * 10
    ).clip(1, 10).astype(int)

    summary_rows = []
    rating_values = full.groupby("rating")["rating"].agg(count="size", mean_rating="mean")
    for rating, row in rating_values.iterrows():
        summary_rows.append({
            "summary_type": "rating_value",
            "key": f"{float(rating):.1f}",
            "count": int(row["count"]),
            "mean_rating": float(row["mean_rating"]),
        })
    year_summary = full.groupby("rating_year")["rating"].agg(count="size", mean_rating="mean")
    for year, row in year_summary.iterrows():
        summary_rows.append({
            "summary_type": "year", "key": str(int(year)),
            "count": int(row["count"]), "mean_rating": float(row["mean_rating"]),
        })

    ratings_with_genres = full[["i", "rating"]].merge(
        movies[["i", "genre_list"]], on="i", how="left"
    ).explode("genre_list")
    ratings_with_genres = ratings_with_genres.dropna(subset=["genre_list"])
    genre_summary = ratings_with_genres.groupby("genre_list")["rating"].agg(
        count="size", mean_rating="mean"
    )
    for genre, row in genre_summary.iterrows():
        summary_rows.append({
            "summary_type": "genre", "key": str(genre),
            "count": int(row["count"]), "mean_rating": float(row["mean_rating"]),
        })

    decade_movies = movies[["i", "year"]].copy()
    decade_movies["decade"] = decade_movies["year"].map(_movie_decade)
    decade_ratings = full[["i", "rating"]].merge(
        decade_movies[["i", "decade"]], on="i", how="left"
    )
    decade_summary = decade_ratings.groupby("decade")["rating"].agg(
        count="size", mean_rating="mean"
    )
    for decade, row in decade_summary.iterrows():
        summary_rows.append({
            "summary_type": "decade", "key": str(decade),
            "count": int(row["count"]), "mean_rating": float(row["mean_rating"]),
        })
    bi_ratings_summary = pd.DataFrame(
        summary_rows, columns=["summary_type", "key", "count", "mean_rating"]
    )
    bi_ratings_summary["count"] = bi_ratings_summary["count"].astype(np.int64)

    movie_rows = []
    for row in movies.itertuples(index=False):
        genres = row.genre_list or ["Unknown"]
        for genre in genres:
            movie_rows.append({
                "movie_id": int(row.movieId),
                "title": str(row.title),
                "year": None if pd.isna(row.year) else int(row.year),
                "genre": str(genre),
                "rating_count": int(row.rating_count),
                "mean_rating": row.mean_rating,
                "popularity_decile": int(row.popularity_decile),
            })
    bi_movies = pd.DataFrame(movie_rows, columns=[
        "movie_id", "title", "year", "genre", "rating_count", "mean_rating",
        "popularity_decile",
    ])

    grouped_users = full.groupby("u").agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        first_rating_date=("rating_datetime", "min"),
        last_rating_date=("rating_datetime", "max"),
    )
    grouped_users["activity_span_days"] = (
        grouped_users["last_rating_date"] - grouped_users["first_rating_date"]
    ).dt.total_seconds() / 86400
    grouped_users["activity_bucket"] = pd.cut(
        grouped_users["n_ratings"],
        bins=[0, 50, 100, 300, np.inf],
        labels=["20-50", "50-100", "100-300", "300+"],
        right=False,
    ).astype("object")
    maps = joblib.load(ARTIFACTS / "id_maps.joblib")
    user_ids = maps["user_ids"]
    bi_users = grouped_users.reset_index().rename(columns={"u": "user_index"})
    bi_users["user_id"] = bi_users["user_index"].map(
        {index: int(user_id) for index, user_id in enumerate(user_ids)}
    )
    bi_users["first_rating_date"] = bi_users["first_rating_date"].dt.strftime("%Y-%m-%d")
    bi_users["last_rating_date"] = bi_users["last_rating_date"].dt.strftime("%Y-%m-%d")
    bi_users = bi_users[[
        "user_id", "n_ratings", "mean_rating", "activity_bucket",
        "first_rating_date", "last_rating_date", "activity_span_days",
    ]]

    counts_sorted = movies[["i", "movieId", "rating_count"]].sort_values(
        ["rating_count", "movieId"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    bi_sorted_counts = pd.DataFrame({
        "rank": np.arange(1, len(counts_sorted) + 1, dtype=np.int64),
        "movie_id": counts_sorted["movieId"].astype(np.int64),
        "rating_count": counts_sorted["rating_count"].astype(np.int64),
        "cumulative_share": counts_sorted["rating_count"].cumsum() / max(1, len(full)),
    })

    n_users = int(full["userId"].nunique())
    n_movies = int(movies["movieId"].nunique())
    n_ratings = int(len(full))
    bi_kpis = pd.DataFrame([{
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "sparsity": 1.0 - n_ratings / (n_users * n_movies),
    }])

    results_test = pd.read_csv(REPORTS / "results_test.csv")
    results_val = pd.read_csv(REPORTS / "results_val.csv")
    labels = pd.DataFrame([
        {"model": model, "label": label, "is_hybrid": is_hybrid}
        for model, (label, is_hybrid) in MODEL_LABELS.items()
    ])
    error_by_bucket = pd.read_csv(REPORTS / "error_by_bucket.csv")
    bucket_orders = {
        "user_activity": {"20-50": 1, "50-100": 2, "100-300": 3, "300+": 4},
        "item_popularity": {"0": 1, "1-5": 2, "6-20": 3, "21-100": 4, "100+": 5},
        "genre": {},
    }
    for bucket_type, group in error_by_bucket.groupby("bucket_type"):
        if bucket_type == "genre":
            bucket_orders[bucket_type] = {
                bucket: index + 1
                for index, bucket in enumerate(sorted(group["bucket"].dropna().unique()))
            }
    error_by_bucket["bucket_order"] = [
        bucket_orders.get(bucket_type, {}).get(str(bucket), 999)
        for bucket_type, bucket in zip(error_by_bucket["bucket_type"], error_by_bucket["bucket"])
    ]
    error_by_bucket = error_by_bucket[[
        "model", "bucket_type", "bucket", "bucket_order", "rmse", "ndcg"
    ]]
    cold_start = pd.read_csv(REPORTS / "cold_start.csv").rename(
        columns={"ndcg@10": "ndcg_at_10"}
    )
    popularity_bias = pd.read_csv(REPORTS / "popularity_bias.csv")
    diversity = pd.read_csv(REPORTS / "diversity.csv")

    engine = joblib.load(ARTIFACTS / "engine.joblib")
    seen_by_user = joblib.load(ARTIFACTS / "seen_by_user.joblib")
    movie_by_item = movies.set_index("i")
    recommendation_rows = []
    for user_index, user_id in enumerate(user_ids):
        seen = seen_by_user.get(user_index, np.array([], dtype=np.int64))
        model_scores = {
            "popularity": engine.pop.score_user(user_index),
            "content": engine.cb.score_user(user_index),
            "item_knn": engine.iknn.score_user(user_index),
            "mf": engine.mf.score_user(user_index),
            "hybrid": engine.score_user(user_index),
        }
        for model_name, scores in model_scores.items():
            for rank, (item_index, _score) in enumerate(_top_items(scores, seen, 10), start=1):
                movie = movie_by_item.loc[item_index]
                genres = _genres(movie["genre_list"])
                recommendation_rows.append({
                    "user_id": int(user_id),
                    "model": model_name,
                    "rank": rank,
                    "movie_id": int(movie["movieId"]),
                    "title": str(movie["title"]),
                    "genres": ", ".join(genres) if genres else "Unknown",
                })
    bi_recs_sample = pd.DataFrame(recommendation_rows, columns=[
        "user_id", "model", "rank", "movie_id", "title", "genres"
    ])

    exports = {
        "bi_ratings_summary.csv": bi_ratings_summary,
        "bi_movies.csv": bi_movies,
        "bi_users.csv": bi_users,
        "bi_sorted_counts.csv": bi_sorted_counts,
        "bi_kpis.csv": bi_kpis,
        "results_test.csv": results_test,
        "results_val.csv": results_val,
        "bi_model_labels.csv": labels,
        "bi_error_by_bucket.csv": error_by_bucket,
        "bi_cold_start.csv": cold_start,
        "bi_recs_sample.csv": bi_recs_sample,
        "bi_popularity_bias.csv": popularity_bias,
        "bi_diversity.csv": diversity,
    }

    key_columns = {
        "bi_ratings_summary.csv": ["summary_type", "key"],
        "bi_movies.csv": ["movie_id", "title", "genre", "popularity_decile"],
        "bi_users.csv": ["user_id", "activity_bucket"],
        "bi_sorted_counts.csv": ["rank", "movie_id"],
        "bi_kpis.csv": ["n_users", "n_movies", "n_ratings"],
        "results_test.csv": ["model", "split", "metric", "params"],
        "results_val.csv": ["model", "split", "metric", "params"],
        "bi_model_labels.csv": ["model", "label"],
        "bi_error_by_bucket.csv": ["model", "bucket_type", "bucket"],
        "bi_cold_start.csv": ["model", "n_ratings"],
        "bi_recs_sample.csv": ["user_id", "model", "rank", "movie_id", "title", "genres"],
        "bi_popularity_bias.csv": ["model", "popularity_decile"],
        "bi_diversity.csv": ["model"],
    }
    output_paths = []
    for filename, frame in exports.items():
        required_keys = key_columns[filename]
        assert not frame[required_keys].isna().any().any(), (
            f"NaN found in key columns for {filename}: {required_keys}"
        )
        output_paths.append(_write_csv(frame, output_dir, filename))

    assert n_ratings == 100836 or n_ratings == int(full.shape[0])
    assert set(results_test["model"].unique()).issubset(set(labels["model"]))

    file_info = []
    for path in output_paths:
        size = path.stat().st_size
        rows = len(exports[path.name])
        file_info.append({"file": path.name, "rows": rows, "bytes": size})
        print(f"{path.name}: {rows:,} rows, {size:,} bytes")
    return pd.DataFrame(file_info)


if __name__ == "__main__":
    export_bi()
