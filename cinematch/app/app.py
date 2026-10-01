"""Flask API and browser app for the frozen CineMatch recommendation engine."""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS = PROJECT_ROOT / "artifacts"
DATA_PROC = PROJECT_ROOT / "data" / "processed"

# Load all serving data once, at module import, not inside request handlers.
movies = pd.read_parquet(DATA_PROC / "movies.parquet").copy()
movies["title_lower"] = movies["title"].fillna("").astype(str).str.lower()
movies_by_id = movies.set_index("movieId", drop=False)
movies_by_item = movies.set_index("i", drop=False)
maps = joblib.load(ARTIFACTS / "id_maps.joblib")
engine = joblib.load(ARTIFACTS / "engine.joblib")
seen_by_user = joblib.load(ARTIFACTS / "seen_by_user.joblib")
movie_stats = pd.read_parquet(DATA_PROC / "movie_stats.parquet").set_index("i")
n_items = len(movies)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def get_n(default=10, cap=30):
	"""Read a bounded positive result count from the query string."""
	try:
		requested = int(request.args.get("n", default))
	except (TypeError, ValueError):
		requested = default
	return max(1, min(requested, cap))


def card(i, score=None):
	"""Serialize one internal item index into the public movie-card shape."""
	item = int(i)
	movie = movies_by_item.loc[item]
	stats = movie_stats.loc[item]
	year = movie.get("year")
	mean_rating = stats.get("mean_rating")
	genre_values = movie.get("genre_list", [])
	if genre_values is None:
		genre_values = []
	result = {
		"movie_id": int(movie["movieId"]),
		"title": str(movie["title"]),
		"genres": ", ".join(str(genre) for genre in genre_values),
		"year": None if pd.isna(year) else int(year),
		"n_ratings": int(stats["n_ratings"]),
		"mean_rating": None if pd.isna(mean_rating) else round(float(mean_rating), 2),
		"score": None if score is None else float(score),
	}
	return result


def top_n(scores, exclude, n):
	"""Return the highest-scoring non-excluded items as movie cards."""
	values = np.asarray(scores, dtype=float).reshape(-1).copy()
	if len(values) != n_items:
		raise ValueError("score vector does not match the movie catalog")
	excluded = np.asarray(list(exclude), dtype=int)
	excluded = excluded[(excluded >= 0) & (excluded < n_items)]
	values[excluded] = -np.inf
	candidates = np.flatnonzero(np.isfinite(values))
	count = min(int(n), len(candidates))
	if count <= 0:
		return []
	selected = candidates[np.argpartition(values[candidates], -count)[-count:]]
	order = np.lexsort((selected, -values[selected]))
	return [card(item, values[item]) for item in selected[order]]


@app.get("/")
def index():
	return render_template("index.html")


@app.get("/api/health")
def health():
	return jsonify({"status": "ok"})


@app.get("/api/search")
def search():
	query = request.args.get("q", "").strip().lower()
	if len(query) < 2:
		return jsonify([])
	matches = movies.loc[movies["title_lower"].str.contains(query, regex=False, na=False)]
	matches = matches.sort_values("movieId").copy()
	matches["n_ratings"] = matches["i"].map(movie_stats["n_ratings"])
	matches = matches.sort_values("n_ratings", ascending=False).head(15)
	return jsonify([card(int(item)) for item in matches["i"]])


@app.get("/api/similar/<int:movie_id>")
def similar(movie_id):
	item = maps["item2idx"].get(movie_id)
	if item is None:
		return jsonify({"error": "unknown movie"}), 404
	indices, scores = engine.similar_items(int(item), n=get_n())
	return jsonify([card(index, score) for index, score in zip(indices, scores)])


@app.get("/api/recommend/user/<int:user_id>")
def recommend_user(user_id):
	user = maps["user2idx"].get(user_id)
	if user is None:
		return jsonify({"error": "unknown user (valid ids are 1 to 610)"}), 404
	excluded = seen_by_user.get(int(user), np.array([], dtype=np.int64))
	return jsonify(top_n(engine.score_user(int(user)), excluded, get_n()))


@app.post("/api/recommend/new")
def recommend_new_user():
	body = request.get_json(silent=True)
	if not isinstance(body, dict) or not isinstance(body.get("ratings"), list):
		return jsonify({"error": "bad rating entry"}), 400
	entries = body["ratings"]
	if len(entries) > 100:
		return jsonify({"error": "bad rating entry"}), 400

	valid_ratings = {}
	item_map = maps["item2idx"]
	for entry in entries:
		if not isinstance(entry, dict):
			return jsonify({"error": "bad rating entry"}), 400
		movie_id = entry.get("movie_id")
		rating = entry.get("rating")
		if isinstance(movie_id, bool) or not isinstance(movie_id, int):
			return jsonify({"error": "bad rating entry"}), 400
		if isinstance(rating, bool) or not isinstance(rating, (int, float)):
			return jsonify({"error": "bad rating entry"}), 400
		rating = float(rating)
		if not np.isfinite(rating):
			return jsonify({"error": "bad rating entry"}), 400
		item = item_map.get(movie_id)
		if item is None or rating < 0.5 or rating > 5.0:
			continue
		valid_ratings[int(item)] = rating

	if len(valid_ratings) < 3:
		return jsonify({"error": "please rate at least 3 known movies"}), 400
	items = np.fromiter(valid_ratings.keys(), dtype=np.int64)
	ratings = np.fromiter(valid_ratings.values(), dtype=float)
	scores = engine.score_new_user(items, ratings)
	return jsonify(top_n(scores, items, get_n()))


@app.get("/api/popular")
def popular():
	count = get_n()
	order = np.argsort(-movie_stats["n_ratings"].to_numpy(), kind="stable")[:count]
	item_indices = movie_stats.index.to_numpy()[order]
	return jsonify([card(item) for item in item_indices])


@app.errorhandler(404)
def not_found(_error):
	return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def internal_error(_error):
	return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
	port = int(os.environ.get("PORT", "5000"))
	app.run(
		host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1",
		port=port,
		debug=os.environ.get("FLASK_DEBUG") == "1",
	)
