import numpy as np
import pandas as pd
import pytest

from src.models.mf import BiasedMF


def _low_rank_ratings(seed=91):
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0, 0.45, (50, 3))
    item_factors = rng.normal(0, 0.45, (40, 3))
    user_bias = rng.normal(0, 0.15, 50)
    item_bias = rng.normal(0, 0.2, 40)
    rows = []
    for user in range(50):
        items = rng.choice(40, size=24, replace=False)
        for item in items:
            rating = (
                3.5
                + user_bias[user]
                + item_bias[item]
                + user_factors[user] @ item_factors[item]
                + rng.normal(0, 0.08)
            )
            rows.append(
                {"u": user, "i": int(item), "rating": float(np.clip(rating, 0.5, 5.0))}
            )
    return pd.DataFrame(rows)


def test_training_rmse_decreases_on_low_rank_data():
    ratings = _low_rank_ratings()
    model = BiasedMF(
        n_factors=8, lr=0.01, reg=0.05, epochs=12, seed=17, verbose=False
    ).fit(ratings, n_users=50, n_items=40)

    assert model.history[-1]["train_rmse"] < model.history[0]["train_rmse"]


def test_predictions_and_user_scores_are_finite_and_sized():
    ratings = _low_rank_ratings()
    model = BiasedMF(
        n_factors=8, lr=0.01, reg=0.05, epochs=5, seed=17, verbose=False
    ).fit(ratings, n_users=50, n_items=40)
    users = np.array([0, 1, 2, 3])
    items = np.array([3, 7, 11, 15])

    predictions = model.predict(users, items)
    scores = model.score_user(0)

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert scores.shape == (40,)
    assert np.isfinite(scores).all()


def test_same_seed_produces_identical_predictions():
    ratings = _low_rank_ratings()
    params = {"n_factors": 8, "lr": 0.01, "epochs": 5, "seed": 23, "verbose": False}
    first = BiasedMF(**params).fit(ratings, 50, 40)
    second = BiasedMF(**params).fit(ratings, 50, 40)
    users = np.array([0, 4, 12, 37])
    items = np.array([2, 9, 18, 31])

    np.testing.assert_array_equal(first.predict(users, items), second.predict(users, items))


def test_huge_learning_rate_raises_non_finite_training_error():
    ratings = _low_rank_ratings()

    with pytest.raises(ValueError, match="learning rate"):
        BiasedMF(
            n_factors=8, lr=10, reg=0.05, epochs=10, seed=17, verbose=False
        ).fit(ratings, n_users=50, n_items=40)