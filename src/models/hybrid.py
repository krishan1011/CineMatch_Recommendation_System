"""Score blending and fold-in utilities for hybrid recommendations."""

from itertools import product

import numpy as np
from scipy.sparse import csr_matrix

from .base import Recommender


def zscore(s):
	"""Standardize a score vector, returning zeros for constant inputs."""
	values = np.asarray(s, dtype=np.float64)
	return (values - values.mean()) / (values.std() + 1e-9)


def score_matrix(model, user_idx):
	"""Build a float32 matrix of per-user z-scored item scores."""
	return np.vstack([zscore(model.score_user(int(user))) for user in user_idx]).astype(
		np.float32
	)


class MatrixScorer(Recommender):
	"""Expose precomputed user-item scores through the recommender interface."""

	name = "matrix_scorer"

	def __init__(self, Z):
		self.Z = np.asarray(Z)

	def fit(self, train_df, n_users, n_items):
		"""Validate the supplied score matrix dimensions."""
		if self.Z.shape != (n_users, n_items):
			raise ValueError("Z must have shape (n_users, n_items)")
		return self

	def predict(self, u, i):
		"""Return precomputed scores for user-item pairs."""
		users = np.asarray(u, dtype=int).ravel()
		items = np.asarray(i, dtype=int).ravel()
		return self.Z[users, items]

	def score_user(self, u):
		"""Return the precomputed score row for user ``u``."""
		return self.Z[int(u)]


def weight_grid(n_models, step=0.1):
	"""Return all non-negative n-model weights summing to one on the step grid."""
	if n_models < 1:
		raise ValueError("n_models must be at least one")
	if step <= 0:
		raise ValueError("step must be positive")
	steps = round(1.0 / step)
	if not np.isclose(steps * step, 1.0):
		raise ValueError("step must divide 1 evenly")
	return [
		tuple(value * step for value in counts)
		for counts in product(range(steps + 1), repeat=n_models)
		if sum(counts) == steps
	]


def fold_in_mf(mf, bias_lambda, items, ratings, lam=1.0):
	"""Fold a new user's ratings into an MF profile and score all items."""
	item_indices = np.asarray(items, dtype=np.int64)
	values = np.asarray(ratings, dtype=np.float64)
	if len(item_indices) != len(values):
		raise ValueError("items and ratings must have equal lengths")
	if len(item_indices) == 0:
		profile = np.zeros(mf.Q.shape[1], dtype=np.float64)
		user_bias = 0.0
	else:
		residuals = values - mf.mu - mf.bi[item_indices]
		user_bias = residuals.sum() / (bias_lambda + len(residuals))
		item_factors = mf.Q[item_indices]
		identity = np.eye(mf.Q.shape[1], dtype=np.float64)
		profile = np.linalg.solve(
			item_factors.T @ item_factors + lam * identity,
			item_factors.T @ (residuals - user_bias),
		)
	return mf.bi + mf.Q @ profile


def fold_in_knn(knn, bias, n_items, items, ratings):
	"""Fold a new user's observed residuals into item-neighborhood scores."""
	item_indices = np.asarray(items, dtype=np.int64)
	values = np.asarray(ratings, dtype=np.float64)
	if len(item_indices) != len(values):
		raise ValueError("items and ratings must have equal lengths")
	residuals = values - bias.mu - bias.bi[item_indices]
	user_bias = residuals.sum() / (15 + len(residuals)) if len(residuals) else 0.0
	evidence = np.zeros(n_items, dtype=np.float64)
	rated = np.zeros(n_items, dtype=np.float64)
	evidence[item_indices] = residuals - user_bias
	rated[item_indices] = 1.0
	numerator = np.asarray(knn.W @ evidence).ravel()
	denominator = np.asarray(knn.Wabs @ rated).ravel() + knn.damp
	return np.divide(
		numerator,
		denominator,
		out=np.zeros(n_items, dtype=np.float64),
		where=denominator != 0,
	)


def fold_in_content(cb, items, ratings):
	"""Build a new-user content profile from centered rating preferences."""
	item_indices = np.asarray(items, dtype=np.int64)
	values = np.asarray(ratings, dtype=np.float64)
	if len(item_indices) != len(values):
		raise ValueError("items and ratings must have equal lengths")
	weights = values - values.mean() if len(values) else np.array([], dtype=float)
	if not len(weights) or np.allclose(weights, 0.0):
		weights = np.ones(len(item_indices), dtype=np.float64)
	profile = np.asarray(cb.X[item_indices].T @ weights).ravel()
	return np.asarray(cb.X @ profile).ravel()


class HybridEngine(Recommender):
	"""Blend fitted model scores for existing and fold-in new users."""

	name = "hybrid"

	def __init__(
		self,
		mf,
		iknn,
		cb,
		pop,
		bias,
		weights,
		n_items,
		weights_cold=None,
		cold_threshold=10,
	):
		self.mf = mf
		self.iknn = iknn
		self.cb = cb
		self.pop = pop
		self.bias = bias
		self.weights = dict(weights)
		self.n_items = n_items
		self.weights_cold = None if weights_cold is None else dict(weights_cold)
		self.cold_threshold = cold_threshold

	def _blend(self, parts, weights):
		"""Blend standardized vectors for the named components with positive weight."""
		scores = np.zeros(self.n_items, dtype=np.float64)
		for name, values in parts.items():
			weight = weights.get(name, 0.0)
			if weight > 0:
				scores += weight * zscore(values)
		return scores

	def score_user(self, u):
		"""Blend fitted MF, item-kNN, content, and popularity scores."""
		parts = {
			"mf": self.mf.score_user(u),
			"iknn": self.iknn.score_user(u),
			"content": self.cb.score_user(u),
			"pop": self.pop.score_user(u),
		}
		return self._blend(parts, self.weights)

	def score_new_user(self, items, ratings):
		"""Blend fold-in personalization with catalog popularity for a new user."""
		item_indices = np.asarray(items, dtype=np.int64)
		values = np.asarray(ratings, dtype=np.float64)
		if len(item_indices) != len(values):
			raise ValueError("items and ratings must have equal lengths")
		if len(item_indices) == 0:
			return zscore(self.pop.score_user(0))

		parts = {
			"mf": fold_in_mf(self.mf, 15, item_indices, values),
			"iknn": fold_in_knn(
				self.iknn, self.bias, self.n_items, item_indices, values
			),
			"content": fold_in_content(self.cb, item_indices, values),
			"pop": self.pop.score_user(0),
		}
		weights = (
			self.weights_cold
			if len(item_indices) < self.cold_threshold and self.weights_cold is not None
			else self.weights
		)
		return self._blend(parts, weights)

	def similar_items(self, i, n=10):
		"""Return nearest items from blended kNN and latent-factor similarities."""
		if n <= 0:
			return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
		item = int(i)
		item_knn_scores = self.iknn.W.getrow(item).toarray().ravel()
		factor_scores = self.mf.Q @ self.mf.Q[item]
		similarities = 0.6 * zscore(item_knn_scores) + 0.4 * zscore(factor_scores)
		similarities[item] = -np.inf
		n_results = min(n, self.n_items - 1)
		if n_results <= 0:
			return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
		candidates = np.argpartition(similarities, -n_results)[-n_results:]
		order = np.lexsort((candidates, -similarities[candidates]))
		indices = candidates[order]
		return indices, similarities[indices]
