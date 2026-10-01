"""Build frozen production artifacts from the complete MovieLens dataset."""

import json

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, DATA_PROC
from src.models.baselines import BiasModel, Popularity
from src.models.content import ContentBased, build_item_features
from src.models.hybrid import HybridEngine
from src.models.knn import ItemKNN
from src.models.mf import BiasedMF


def build_artifacts():
    """Fit configured serving models on all supplied split data and persist them."""
    train = pd.read_parquet(DATA_PROC / "train.parquet")
    val = pd.read_parquet(DATA_PROC / "val.parquet")
    test = pd.read_parquet(DATA_PROC / "test.parquet")
    movies = pd.read_parquet(DATA_PROC / "movies.parquet")
    full = pd.concat([train, val, test], ignore_index=True)
    n_users = int(full["u"].max()) + 1
    n_items = len(movies)

    content_config = json.loads((ARTIFACTS / "content_config.json").read_text())
    knn_config = json.loads((ARTIFACTS / "knn_config.json").read_text())
    mf_config = json.loads((ARTIFACTS / "mf_config.json").read_text())
    hybrid_config = json.loads((ARTIFACTS / "hybrid_config.json").read_text())

    bias = BiasModel().fit(full, n_users, n_items)
    popularity = Popularity("count").fit(full, n_users, n_items)
    features, _ = build_item_features(
        movies,
        w_genre=content_config["w_genre"],
        w_tag=content_config["w_tag"],
        w_decade=content_config["w_decade"],
        use_genre=content_config["use_genre"],
        use_tag=content_config["use_tag"],
        use_decade=content_config["use_decade"],
    )
    content = ContentBased(
        features,
        profile_mode=content_config["profile_mode"],
        popularity_weight=content_config["popularity_weight"],
    ).fit(full, n_users, n_items)

    item_config = knn_config["item_knn"]["ndcg_best"]
    item_knn = ItemKNN(
        bias,
        k=item_config["k"],
        shrink=item_config["shrink"],
        damp=item_config["damp"],
        mode="residual",
    ).fit(full, n_users, n_items)

    mf_saved = mf_config["mf_ndcg"]
    mf_parameters = {
        key: mf_saved[key]
        for key in ("n_factors", "lr", "reg", "init_std", "seed", "patience")
    }
    mf = BiasedMF(
        **mf_parameters,
        epochs=int(mf_saved["best_epoch"]),
        verbose=False,
    ).fit(full, n_users, n_items)

    engine = HybridEngine(
        mf,
        item_knn,
        content,
        popularity,
        bias,
        hybrid_config["weights"],
        n_items,
        weights_cold=hybrid_config["weights_cold"],
        cold_threshold=hybrid_config["cold_threshold"],
    )
    seen_by_user = {
        int(user): group["i"].to_numpy(dtype=np.int64)
        for user, group in full.groupby("u", sort=False)
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    engine_path = ARTIFACTS / "engine.joblib"
    seen_path = ARTIFACTS / "seen_by_user.joblib"
    joblib.dump(engine, engine_path, compress=3)
    joblib.dump(seen_by_user, seen_path, compress=3)

    stats = full.groupby("i")["rating"].agg(
        n_ratings="size", mean_rating="mean"
    )
    movie_stats = movies[["i", "movieId"]].merge(
        stats, left_on="i", right_index=True, how="left"
    )
    movie_stats["n_ratings"] = movie_stats["n_ratings"].fillna(0).astype(np.int64)
    movie_stats = movie_stats[["i", "movieId", "n_ratings", "mean_rating"]]
    movie_stats_path = DATA_PROC / "movie_stats.parquet"
    movie_stats.to_parquet(movie_stats_path, index=False)

    artifact_paths = [engine_path, seen_path, movie_stats_path]
    sizes_mb = {path.name: path.stat().st_size / (1024 * 1024) for path in artifact_paths}
    total_mb = sum(sizes_mb.values())
    for name, size_mb in sizes_mb.items():
        print(f"{name}: {size_mb:.2f} MB")
    print(f"Total production artifacts: {total_mb:.2f} MB")
    assert total_mb < 50, f"Production artifacts total {total_mb:.2f} MB, exceeding 50 MB"
    return {"engine": engine_path, "seen_by_user": seen_path, "movie_stats": movie_stats_path}


if __name__ == "__main__":
    build_artifacts()
