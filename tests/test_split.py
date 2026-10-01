import pandas as pd
import pytest
from pathlib import Path

from src.config import DATA_PROC
from src.data import clean_ratings, make_splits, temporal_split_per_user

REAL_SPLIT_FILES = [
    DATA_PROC / "train.parquet",
    DATA_PROC / "val.parquet",
    DATA_PROC / "test.parquet",
]


REAL_SPLIT_FILES = [
    DATA_PROC / "train.parquet",
    DATA_PROC / "val.parquet",
    DATA_PROC / "test.parquet",
]


def _synthetic_ratings():
    rows = []
    for user_id in (1, 2):
        for offset in range(10):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": user_id * 100 + offset,
                    "rating": 3.0 + offset / 10,
                    "timestamp": 1_000 + offset,
                }
            )
    return pd.DataFrame(rows)


def _pairs(frame):
    return set(zip(frame["userId"], frame["movieId"]))


def test_temporal_split_per_user_uses_latest_fraction_without_leakage():
    ratings = _synthetic_ratings()

    earlier, later = temporal_split_per_user(ratings, 0.2)

    assert len(earlier) == 16
    assert len(later) == 4
    assert _pairs(earlier).isdisjoint(_pairs(later))
    for user_id in ratings["userId"].unique():
        user_earlier = earlier.loc[earlier["userId"] == user_id]
        user_later = later.loc[later["userId"] == user_id]
        assert len(user_earlier) == 8
        assert len(user_later) == 2
        assert user_earlier["timestamp"].max() <= user_later["timestamp"].min()


def test_make_splits_are_disjoint_ordered_and_cover_every_user():
    ratings = _synthetic_ratings()

    train, val, test = make_splits(ratings)

    assert (len(train), len(val), len(test)) == (14, 2, 4)
    assert len(train) + len(val) + len(test) == len(ratings)
    assert _pairs(train).isdisjoint(_pairs(val))
    assert _pairs(train).isdisjoint(_pairs(test))
    assert _pairs(val).isdisjoint(_pairs(test))
    for user_id in ratings["userId"].unique():
        user_train = train.loc[train["userId"] == user_id]
        user_val = val.loc[val["userId"] == user_id]
        user_test = test.loc[test["userId"] == user_id]
        assert not user_train.empty
        assert not user_val.empty
        assert not user_test.empty
        assert user_train["timestamp"].max() <= user_val["timestamp"].min()
        assert user_val["timestamp"].max() <= user_test["timestamp"].min()


def test_clean_ratings_keeps_latest_duplicate():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 10, 20],
            "rating": [2.0, 4.5, 3.0],
            "timestamp": [100, 300, 200],
        }
    )

    cleaned = clean_ratings(ratings)

    duplicate = cleaned.loc[
        (cleaned["userId"] == 1) & (cleaned["movieId"] == 10)
    ].iloc[0]
    assert len(cleaned) == 2
    assert duplicate["rating"] == 4.5
    assert duplicate["timestamp"] == 300
    assert cleaned["timestamp"].is_monotonic_increasing


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_SPLIT_FILES),
    reason="real processed split parquet files are unavailable",
)
def test_no_future_leakage():
    train, val, test = [pd.read_parquet(path) for path in REAL_SPLIT_FILES]
    bounds = pd.DataFrame({
        "train_max": train.groupby("u")["timestamp"].max(),
        "val_min": val.groupby("u")["timestamp"].min(),
        "test_min": test.groupby("u")["timestamp"].min(),
    })

    assert bounds.notna().all().all()
    assert (bounds["train_max"] <= bounds["val_min"]).all()
    assert (bounds["val_min"] <= bounds["test_min"]).all()


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_SPLIT_FILES),
    reason="real processed split parquet files are unavailable",
)
def test_no_overlap():
    train, val, test = [pd.read_parquet(path) for path in REAL_SPLIT_FILES]
    pairs = [
        set(zip(frame["u"], frame["i"]))
        for frame in (train, val, test)
    ]

    assert pairs[0].isdisjoint(pairs[1])
    assert pairs[0].isdisjoint(pairs[2])
    assert pairs[1].isdisjoint(pairs[2])


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_SPLIT_FILES),
    reason="real processed split parquet files are not available",
)
def test_no_future_leakage():
    train, val, test = [pd.read_parquet(path) for path in REAL_SPLIT_FILES]
    bounds = pd.DataFrame({
        "train_max": train.groupby("u")["timestamp"].max(),
        "val_min": val.groupby("u")["timestamp"].min(),
        "test_min": test.groupby("u")["timestamp"].min(),
    })

    assert bounds.notna().all().all()
    assert (bounds["train_max"] <= bounds["val_min"]).all()
    assert (bounds["val_min"] <= bounds["test_min"]).all()


@pytest.mark.skipif(
    not all(path.exists() for path in REAL_SPLIT_FILES),
    reason="real processed split parquet files are not available",
)
def test_no_overlap():
    train, val, test = [pd.read_parquet(path) for path in REAL_SPLIT_FILES]
    pairs = [
        set(zip(frame["u"], frame["i"]))
        for frame in (train, val, test)
    ]

    assert pairs[0].isdisjoint(pairs[1])
    assert pairs[0].isdisjoint(pairs[2])
    assert pairs[1].isdisjoint(pairs[2])
