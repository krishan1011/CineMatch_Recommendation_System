import joblib
import numpy as np
from scipy.sparse import csr_matrix

from src.models.hybrid import HybridEngine, fold_in_mf, weight_grid, zscore


class _ToyMF:
    def __init__(self):
        self.mu = 3.0
        self.bi = np.array([0.1, -0.1, 0.0, 0.2, -0.2])
        self.Q = np.arange(10, dtype=float).reshape(5, 2) / 10

    def score_user(self, user):
        return self.bi + self.Q @ np.array([0.2, 0.3])


class _ToyKNN:
    def __init__(self):
        self.W = csr_matrix(np.array(
            [[0, 0.5, 0, 0, 0], [0.5, 0, 0.2, 0, 0], [0, 0.2, 0, 0.4, 0],
             [0, 0, 0.4, 0, 0.3], [0, 0, 0, 0.3, 0]], dtype=float
        ))
        self.Wabs = abs(self.W)
        self.damp = 1.0

    def score_user(self, user):
        return np.arange(5, dtype=float)


class _ToyContent:
    def __init__(self):
        self.X = csr_matrix(np.eye(5, dtype=float))

    def score_user(self, user):
        return np.arange(5, dtype=float) * 2


class _ToyPopularity:
    def score_user(self, user):
        return np.array([1.0, 3.0, 2.0, 4.0, 0.0])


class _ToyBias:
    mu = 3.0
    bi = np.array([0.1, -0.1, 0.0, 0.2, -0.2])


def _engine():
    return HybridEngine(
        _ToyMF(),
        _ToyKNN(),
        _ToyContent(),
        _ToyPopularity(),
        _ToyBias(),
        {"mf": 0.25, "iknn": 0.25, "content": 0.25, "pop": 0.25},
        n_items=5,
    )


def test_zscore_constant_vector_is_zero():
    assert np.array_equal(zscore(np.ones(5)), np.zeros(5))


def test_weight_grid_sums_to_one():
    weights = weight_grid(4, step=0.1)

    assert weights
    assert all(np.isclose(sum(values), 1.0) for values in weights)


def test_fold_in_mf_returns_finite_item_scores():
    model = _ToyMF()

    scores = fold_in_mf(model, bias_lambda=15, items=[0, 2], ratings=[4.0, 2.0])

    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


def test_empty_new_user_is_zscored_popularity():
    engine = _engine()

    actual = engine.score_new_user([], [])

    np.testing.assert_allclose(actual, zscore(_ToyPopularity().score_user(0)))


def test_engine_survives_joblib_round_trip(tmp_path):
    engine = _engine()
    path = tmp_path / "hybrid.joblib"
    joblib.dump(engine, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(restored.score_user(1), engine.score_user(1))


def test_similar_items_excludes_query_item():
    indices, scores = _engine().similar_items(2, n=4)

    assert len(indices) == 4
    assert len(scores) == 4
    assert 2 not in indices
