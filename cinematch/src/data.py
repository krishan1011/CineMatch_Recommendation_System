# src/data.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from math import ceil

from .config import DATA_RAW, TEST_FRAC, VAL_FRAC


def load_raw():
    raw_dir = DATA_RAW if DATA_RAW.exists() else DATA_RAW.parent
    ratings = pd.read_csv(raw_dir / "ratings.csv")
    movies = pd.read_csv(raw_dir / "movies.csv")
    tags = pd.read_csv(raw_dir / "tags.csv")
    links = pd.read_csv(raw_dir / "links.csv")
    return ratings, movies, tags, links


def clean_ratings(r):
    """Sort ratings by time and retain the latest row per user-movie pair."""
    return (
        r.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(["userId", "movieId"], keep="last")
        .reset_index(drop=True)
    )


def clean_movies(movies):
    """Add parsed release years, normalized titles, and genre lists."""
    cleaned = movies.copy()
    cleaned["year"] = cleaned["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype(float)
    cleaned["clean_title"] = cleaned["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    ).str.strip()
    cleaned["genre_list"] = cleaned["genres"].map(
        lambda genres: []
        if pd.isna(genres) or genres == "(no genres listed)"
        else genres.split("|")
    )
    return cleaned


def aggregate_tags(tags):
    """Normalize tags and join each movie's tags into a space-separated string."""
    normalized = tags.dropna(subset=["tag"]).copy()
    normalized["tag"] = normalized["tag"].str.lower().str.strip()
    tag_text = normalized.groupby("movieId")["tag"].agg(" ".join)
    tag_text.name = "tag_text"
    return tag_text


def temporal_split_per_user(df, frac):
    """Split each user's latest ceil(frac * n) rows into a later partition."""
    ordered = df.sort_values(["userId", "timestamp"], kind="mergesort").copy()
    group = ordered.groupby("userId", sort=False)
    n = group["timestamp"].transform("size")
    pos_from_end = group.cumcount(ascending=False)
    n_later = np.ceil(n * frac).astype(int)
    later_mask = pos_from_end < n_later
    earlier = ordered.loc[~later_mask].reset_index(drop=True)
    later = ordered.loc[later_mask].reset_index(drop=True)
    return earlier, later


def make_splits(ratings):
    """Create per-user chronological train, validation, and test partitions."""
    train_val, test = temporal_split_per_user(ratings, TEST_FRAC)
    train, val = temporal_split_per_user(train_val, VAL_FRAC)
    return train, val, test


def build_index_maps(ratings, movies):
    """Build sorted user and full-catalog item IDs with contiguous indices."""
    user_ids = sorted(ratings["userId"].unique())
    item_ids = sorted(movies["movieId"].unique())
    user2idx = {user_id: index for index, user_id in enumerate(user_ids)}
    item2idx = {item_id: index for index, item_id in enumerate(item_ids)}
    return user_ids, item_ids, user2idx, item2idx


def add_indices(df, user2idx, item2idx):
    """Add int32 user and item index columns using the supplied ID maps."""
    indexed = df.copy()
    indexed["u"] = indexed["userId"].map(user2idx).astype(np.int32)
    indexed["i"] = indexed["movieId"].map(item2idx).astype(np.int32)
    return indexed


def to_csr(df, n_users, n_items, value_col="rating"):
    """Convert indexed interactions into a float32 SciPy CSR matrix."""
    return csr_matrix(
        (
            df[value_col].to_numpy(dtype=np.float32),
            (df["u"].to_numpy(), df["i"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
