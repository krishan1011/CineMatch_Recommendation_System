import numpy as np
import pandas as pd

from src.models.content import ContentBased, build_item_features


def _movies():
    return pd.DataFrame(
        {
            "genre_list": [
                ["Adventure", "Children"],
                ["Adventure"],
                ["Drama"],
                [],
            ],
            "tag_text": [
                "toy story animation",
                "toy story pixar",
                "dark knight batman",
                "",
            ],
            "year": [1995.0, 1995.0, 2008.0, np.nan],
        }
    )


def test_item_feature_rows_are_unit_or_zero_norm():
    features, _ = build_item_features(
        _movies(), use_tag=False, use_decade=False
    )
    norms = np.sqrt(features.multiply(features).sum(axis=1).A1)

    assert np.allclose(norms[:3], 1.0)
    assert norms[3] == 0.0


def test_content_scores_have_one_value_per_item():
    movies = _movies()
    features, _ = build_item_features(movies)
    train = pd.DataFrame(
        {"u": [0, 0, 1], "i": [0, 1, 2], "rating": [5.0, 4.0, 3.0]}
    )
    model = ContentBased(features).fit(train, n_users=2, n_items=len(movies))

    assert model.score_user(0).shape == (len(movies),)


def test_similar_items_never_returns_query_item():
    movies = _movies()
    features, _ = build_item_features(movies)
    model = ContentBased(features).fit(
        pd.DataFrame({"u": [0], "i": [0], "rating": [5.0]}),
        n_users=1,
        n_items=len(movies),
    )

    indices, scores = model.similar_items(0, n=3)

    assert len(indices) == 3
    assert len(scores) == 3
    assert 0 not in indices


def test_tiny_movie_frame_builds_feature_blocks():
    features, vectorizers = build_item_features(_movies())

    assert features.shape[0] == 4
    assert set(vectorizers) == {"genre", "tag", "decade"}