from pathlib import Path

import pytest

pytest.importorskip("flask")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "artifacts" / "engine.joblib").exists():
    pytest.skip("production artifacts have not been built", allow_module_level=True)

from app.app import app, maps, seen_by_user


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_search_returns_toy_titles(client):
    response = client.get("/api/search?q=toy")

    assert response.status_code == 200
    assert response.get_json()


def test_similar_returns_ten_and_excludes_query_movie(client):
    response = client.get("/api/similar/1")

    assert response.status_code == 200
    cards = response.get_json()
    assert len(cards) == 10
    assert all(card["movie_id"] != 1 for card in cards)


def test_unknown_similar_movie_is_404(client):
    response = client.get("/api/similar/999999")

    assert response.status_code == 404
    assert response.get_json() == {"error": "unknown movie"}


def test_existing_user_recommendations_exclude_rated_movies(client):
    response = client.get("/api/recommend/user/1")

    assert response.status_code == 200
    cards = response.get_json()
    assert len(cards) == 10
    user_index = maps["user2idx"][1]
    seen_items = set(seen_by_user[user_index])
    seen_movie_ids = {
        movie_id for movie_id, item_index in maps["item2idx"].items()
        if item_index in seen_items
    }
    assert all(card["movie_id"] not in seen_movie_ids for card in cards)


def test_unknown_user_is_404(client):
    response = client.get("/api/recommend/user/99999")

    assert response.status_code == 404
    assert response.get_json() == {"error": "unknown user (valid ids are 1 to 610)"}


def test_new_user_requires_at_least_three_valid_ratings(client):
    popular = client.get("/api/popular?n=2").get_json()
    response = client.post(
        "/api/recommend/new?n=10",
        json={"ratings": [
            {"movie_id": card["movie_id"], "rating": 4.0}
            for card in popular
        ]},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "please rate at least 3 known movies"}


def test_new_user_recommendations_validate_and_exclude_rated_movies(client):
    rated = client.get("/api/popular?n=6").get_json()
    response = client.post(
        "/api/recommend/new?n=10",
        json={"ratings": [
            {"movie_id": card["movie_id"], "rating": 4.5}
            for card in rated
        ]},
    )

    assert response.status_code == 200
    recommendations = response.get_json()
    assert len(recommendations) == 10
    rated_ids = {card["movie_id"] for card in rated}
    assert all(card["movie_id"] not in rated_ids for card in recommendations)


def test_new_user_malformed_body_is_400(client):
    response = client.post("/api/recommend/new", json={"ratings": ["bad"]})

    assert response.status_code == 400
    assert response.get_json() == {"error": "bad rating entry"}


def test_n_is_capped_at_thirty(client):
    response = client.get("/api/popular?n=999")

    assert response.status_code == 200
    assert len(response.get_json()) == 30
