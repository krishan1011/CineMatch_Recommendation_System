from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.baselines import BiasModel
from src.models.knn import ItemKNN, UserKNN, residual_matrix, topk_similarity


def _synthetic_ratings():
    rng = np.random.default_rng(73)
    rows = []
    for user in range(20):
        items = rng.choice(30, size=10, replace=False)
        ratings = rng.integers(1, 6, size=len(items))
        rows.extend(
            {"u": user, "i": int(item), "rating": float(rating)}
            for item, rating in zip(items, ratings)
        )
    return pd.DataFrame(rows)


def test_topk_similarity_is_sparse_nonnegative_and_bounded_per_row():
    rng = np.random.default_rng(73)
    dense = (rng.random((20, 30)) < 0.3).astype(np.float32)
    matrix = csr_matrix(dense * rng.normal(size=(20, 30)).astype(np.float32))
    observed = csr_matrix(dense)

    similarity = topk_similarity(matrix, observed, k=4, shrink=10, chunk=6)

    assert similarity.shape == (20, 20)
    assert similarity.nnz > 0
    assert np.all(similarity.diagonal() == 0)
    assert np.all(similarity.data >= 0)
    assert np.all(np.diff(similarity.indptr) <= 4)


def test_shrinkage_reduces_low_overlap_similarity():
    matrix = csr_matrix([[1, 1, 0], [1, 0, 1]], dtype=np.float32)
    observed = (matrix != 0).astype(np.float32)

    unshrunk = topk_similarity(matrix, observed, k=1, shrink=0)
    shrunk = topk_similarity(matrix, observed, k=1, shrink=100)

    assert unshrunk[0, 1] > shrunk[0, 1]


def test_item_and_user_knn_score_and_predict_are_finite():
    train = _synthetic_ratings()
    bias = BiasModel().fit(train, n_users=20, n_items=30)

    for model in (
        ItemKNN(bias, k=5, shrink=2).fit(train, 20, 30),
        UserKNN(bias, k=5, shrink=2).fit(train, 20, 30),
        ItemKNN(bias, k=5, shrink=2, mode="implicit").fit(train, 20, 30),
        UserKNN(bias, k=5, shrink=2, mode="implicit").fit(train, 20, 30),
    ):
        scores = model.score_user(0)
        predictions = model.predict(np.array([0, 1, 2]), np.array([1, 2, 3]))
        assert scores.shape == (30,)
        assert np.isfinite(scores).all()
        assert predictions.shape == (3,)
        assert np.isfinite(predictions).all()


def test_residuals_are_zero_for_user_at_bias_baseline():
    train = pd.DataFrame(
        {
            "u": [0, 0, 0, 1],
            "i": [0, 1, 2, 0],
            "rating": [3.5, 2.5, 4.0, 3.0],
        }
    )
    bias = SimpleNamespace(
        mu=3.0,
        bu=np.array([0.0, 0.0]),
        bi=np.array([0.5, -0.5, 1.0]),
        n_users=2,
        n_items=3,
    )
    train.loc[train["u"] == 0, "rating"] = (
        bias.mu + bias.bu[0] + bias.bi[train.loc[train["u"] == 0, "i"]]
    )

    residuals, observed = residual_matrix(train, bias)

    assert residuals.getrow(0).nnz == 0
    assert observed.getrow(0).nnz == 3