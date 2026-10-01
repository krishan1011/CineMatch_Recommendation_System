"""Sparse item-feature construction and content-based recommendations."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import Recommender


def build_item_features(
	movies,
	w_genre=1.0,
	w_tag=0.5,
	w_decade=0.3,
	use_genre=True,
	use_tag=True,
	use_decade=True,
):
	"""Build weighted, row-normalized sparse genre, tag, and decade features."""
	blocks = []
	vectorizers = {}

	def fit_or_empty(vectorizer, documents):
		try:
			return vectorizer.fit_transform(documents)
		except ValueError as error:
			if "empty vocabulary" not in str(error).lower() and "no terms remain" not in str(error).lower():
				raise
			return csr_matrix((len(movies), 0), dtype=float)

	if use_genre:
		genre_text = movies["genre_list"].map(
			lambda genres: "|".join(genres) if genres is not None else ""
		)
		genre_vectorizer = TfidfVectorizer(
			tokenizer=lambda text: [token for token in text.split("|") if token],
			lowercase=False,
			token_pattern=None,
		)
		blocks.append(fit_or_empty(genre_vectorizer, genre_text) * w_genre)
		vectorizers["genre"] = genre_vectorizer

	if use_tag:
		tag_text = movies["tag_text"].fillna("").astype(str)
		tag_vectorizer = TfidfVectorizer(
			stop_words="english", min_df=2, max_features=2000
		)
		blocks.append(fit_or_empty(tag_vectorizer, tag_text) * w_tag)
		vectorizers["tag"] = tag_vectorizer

	if use_decade:
		def decade_token(year):
			if pd.isna(year):
				return "decade_unknown"
			return f"decade_{int(float(year)) // 10 * 10}"

		decade_text = movies["year"].map(decade_token)
		decade_vectorizer = TfidfVectorizer(
			tokenizer=lambda text: [text], lowercase=False, token_pattern=None
		)
		blocks.append(decade_vectorizer.fit_transform(decade_text) * w_decade)
		vectorizers["decade"] = decade_vectorizer

	if not blocks:
		raise ValueError("At least one item feature block must be enabled")

	features = hstack(blocks, format="csr")
	features = normalize(features, norm="l2", axis=1).tocsr()
	return features, vectorizers


class ContentBased(Recommender):
	"""Recommend items using content similarity to a user's rated items."""

	name = "content"

	def __init__(self, X, profile_mode="centered", popularity_weight=0.0):
		if profile_mode not in {"centered", "ones", "liked"}:
			raise ValueError("profile_mode must be 'centered', 'ones', or 'liked'")
		self.X = csr_matrix(X)
		self.profile_mode = profile_mode
		self.popularity_weight = float(popularity_weight)

	def fit(self, train, n_users, n_items):
		"""Build train-only user histories, rating means, and popularity stats."""
		users = train["u"].to_numpy(dtype=int)
		items = train["i"].to_numpy(dtype=int)
		ratings = train["rating"].to_numpy(dtype=float)
		if self.X.shape[0] != n_items:
			raise ValueError("feature matrix row count must equal n_items")

		self.n_users = n_users
		self.n_items = n_items
		self.global_mean = float(ratings.mean()) if len(ratings) else 0.0
		self.user_items = csr_matrix(
			(ratings, (users, items)), shape=(n_users, n_items), dtype=float
		)
		user_counts = np.bincount(users, minlength=n_users)
		user_sums = np.bincount(users, weights=ratings, minlength=n_users)
		self.user_means = np.divide(
			user_sums,
			user_counts,
			out=np.full(n_users, self.global_mean, dtype=float),
			where=user_counts > 0,
		)

		item_counts = np.bincount(items, minlength=n_items).astype(float)
		self.item_popularity = (
			item_counts / len(train) if len(train) else np.zeros(n_items)
		)
		count_std = item_counts.std()
		self.item_popularity_z = (
			(item_counts - item_counts.mean()) / count_std
			if count_std > 0
			else np.zeros(n_items, dtype=float)
		)
		return self

	def profile(self, u):
		"""Build a content profile from a user's train-only item history."""
		history = self.user_items.getrow(int(u))
		item_indices = history.indices
		ratings = history.data
		if self.profile_mode == "ones":
			weights = np.ones(len(item_indices), dtype=float)
		elif self.profile_mode == "liked":
			weights = (ratings >= 4.0).astype(float)
		else:
			weights = ratings - self.user_means[int(u)]

		if not len(weights) or np.allclose(weights, 0.0):
			weights = np.ones(len(item_indices), dtype=float)
		profile = self.X[item_indices].T @ weights
		return np.asarray(profile).reshape(-1)

	def score_user(self, u):
		"""Return content similarity plus weighted train-popularity scores."""
		scores = np.asarray(self.X @ self.profile(u)).reshape(-1)
		return scores + self.popularity_weight * self.item_popularity_z

	def predict(self, u, i):
		"""Return the fitted train mean for each user's rating prediction."""
		users = np.asarray(u, dtype=int)
		return self.user_means[users]

	def similar_items(self, i, n=10):
		"""Return up to ``n`` nearest other item indices and cosine scores."""
		if n <= 0:
			return np.array([], dtype=int), np.array([], dtype=float)
		item = int(i)
		similarities = (self.X @ self.X[item].T).toarray().reshape(-1)
		similarities[item] = -np.inf
		n_results = min(n, self.n_items - 1)
		if n_results <= 0:
			return np.array([], dtype=int), np.array([], dtype=float)
		candidates = np.argpartition(similarities, -n_results)[-n_results:]
		order = np.lexsort((candidates, -similarities[candidates]))
		indices = candidates[order]
		return indices, similarities[indices]
