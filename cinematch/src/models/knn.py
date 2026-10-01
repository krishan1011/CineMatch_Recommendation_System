"""Memory-bounded neighborhood collaborative filtering recommenders."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .base import Recommender


def residual_matrix(train, bias):
	"""Build float32 rating residuals and a binary observed-rating matrix."""
	users = train["u"].to_numpy(dtype=np.int32)
	items = train["i"].to_numpy(dtype=np.int32)
	ratings = train["rating"].to_numpy(dtype=np.float32)
	residuals = ratings - (
		bias.mu + bias.bu[users] + bias.bi[items]
	).astype(np.float32)
	shape = (bias.n_users, bias.n_items)
	residual = csr_matrix((residuals, (users, items)), shape=shape, dtype=np.float32)
	residual.eliminate_zeros()
	observed = csr_matrix(
		(np.ones(len(train), dtype=np.float32), (users, items)),
		shape=shape,
		dtype=np.float32,
	)
	return residual, observed


def topk_similarity(M, B, k=50, shrink=25, chunk=1000):
	"""Compute row-wise shrunk cosine neighbors using dense chunk buffers only."""
	if k < 0:
		raise ValueError("k must be non-negative")
	if shrink < 0:
		raise ValueError("shrink must be non-negative")
	if chunk <= 0:
		raise ValueError("chunk must be positive")

	matrix = csr_matrix(M, dtype=np.float32)
	observed = csr_matrix(B, dtype=np.float32)
	if matrix.shape != observed.shape:
		raise ValueError("M and B must have the same shape")

	n_rows = matrix.shape[0]
	kk = min(int(k), max(0, n_rows - 1))
	if kk == 0 or n_rows == 0:
		return csr_matrix((n_rows, n_rows), dtype=np.float32)

	normalized = normalize(matrix, norm="l2", axis=1).tocsr()
	output_rows = []
	output_cols = []
	output_values = []

	for start in range(0, n_rows, chunk):
		end = min(start + chunk, n_rows)
		similarities = (normalized[start:end] @ normalized.T).toarray()
		common_counts = (observed[start:end] @ observed.T).toarray()
		shrink_factors = np.divide(
			common_counts,
			common_counts + shrink,
			out=np.zeros_like(common_counts, dtype=np.float32),
			where=(common_counts + shrink) != 0,
		)
		similarities *= shrink_factors

		for local_row, global_row in enumerate(range(start, end)):
			similarities[local_row, global_row] = 0.0
			np.maximum(similarities[local_row], 0.0, out=similarities[local_row])
			candidates = np.flatnonzero(similarities[local_row] > 0)
			if len(candidates) > kk:
				chosen = np.argpartition(
					similarities[local_row, candidates], -kk
				)[-kk:]
				candidates = candidates[chosen]
			values = similarities[local_row, candidates]
			nonzero = values > 0
			output_rows.extend([global_row] * int(nonzero.sum()))
			output_cols.extend(candidates[nonzero].tolist())
			output_values.extend(values[nonzero].tolist())

	return csr_matrix(
		(np.asarray(output_values, dtype=np.float32), (output_rows, output_cols)),
		shape=(n_rows, n_rows),
		dtype=np.float32,
	)


class ItemKNN(Recommender):
	"""Item-neighborhood recommender using residual or implicit similarities."""

	name = "item_knn"

	def __init__(self, bias, k=50, shrink=25, damp=1.0, mode="residual"):
		if mode not in {"residual", "implicit"}:
			raise ValueError("mode must be 'residual' or 'implicit'")
		self.bias = bias
		self.k = k
		self.shrink = shrink
		self.damp = damp
		self.mode = mode
		self._predict_cache = {}

	def fit(self, train, n_users, n_items):
		"""Fit item similarities and residual/implicit histories on train only."""
		self.n_users = n_users
		self.n_items = n_items
		self.E, self.B = residual_matrix(train, self.bias)
		self.W = topk_similarity(
			self.E.T.tocsr(), self.B.T.tocsr(), self.k, self.shrink
		)
		self.Wabs = abs(self.W).tocsr()
		if self.mode == "implicit":
			liked = train["rating"].to_numpy() >= 4.0
			self.Bin = csr_matrix(
				(
					np.ones(int(liked.sum()), dtype=np.float32),
					(
						train.loc[liked, "u"].to_numpy(dtype=np.int32),
						train.loc[liked, "i"].to_numpy(dtype=np.int32),
					),
				),
				shape=(n_users, n_items),
				dtype=np.float32,
			)
		self._predict_cache = {}
		return self

	def _residual_scores(self, u):
		user_residuals = self.E.getrow(u).toarray().ravel()
		user_observed = self.B.getrow(u).toarray().ravel()
		numerator = np.asarray(self.W @ user_residuals).ravel()
		denominator = np.asarray(self.Wabs @ user_observed).ravel() + self.damp
		return np.divide(
			numerator,
			denominator,
			out=np.zeros(self.n_items, dtype=np.float32),
			where=denominator != 0,
		)

	def score_user(self, u):
		"""Score all items from residual neighbors or liked-item neighbors."""
		user = int(u)
		if self.mode == "implicit":
			return np.asarray(self.W @ self.Bin.getrow(user).toarray().ravel()).ravel()
		return self._residual_scores(user)

	def predict(self, u, i):
		"""Predict bias baseline plus residual neighbor correction for pairs."""
		users = np.asarray(u, dtype=int).ravel()
		items = np.asarray(i, dtype=int).ravel()
		residual = np.empty(len(users), dtype=np.float32)
		for user in np.unique(users):
			if user not in self._predict_cache:
				self._predict_cache[user] = self._residual_scores(int(user))
			residual[users == user] = self._predict_cache[user][items[users == user]]
		return (
			self.bias.mu
			+ self.bias.bu[users]
			+ self.bias.bi[items]
			+ residual
		)


class UserKNN(Recommender):
	"""User-neighborhood recommender using residual or implicit similarities."""

	name = "user_knn"

	def __init__(self, bias, k=40, shrink=10, damp=1.0, mode="residual"):
		if mode not in {"residual", "implicit"}:
			raise ValueError("mode must be 'residual' or 'implicit'")
		self.bias = bias
		self.k = k
		self.shrink = shrink
		self.damp = damp
		self.mode = mode
		self._predict_cache = {}

	def fit(self, train, n_users, n_items):
		"""Fit user similarities and residual/implicit histories on train only."""
		self.n_users = n_users
		self.n_items = n_items
		self.E, self.B = residual_matrix(train, self.bias)
		self.W = topk_similarity(self.E, self.B, self.k, self.shrink)
		self.Wabs = abs(self.W).tocsr()
		if self.mode == "implicit":
			liked = train["rating"].to_numpy() >= 4.0
			self.Bin = csr_matrix(
				(
					np.ones(int(liked.sum()), dtype=np.float32),
					(
						train.loc[liked, "u"].to_numpy(dtype=np.int32),
						train.loc[liked, "i"].to_numpy(dtype=np.int32),
					),
				),
				shape=(n_users, n_items),
				dtype=np.float32,
			)
		self._predict_cache = {}
		return self

	def _num_den(self, u):
		"""Return item-wise weighted residual sums and observed-weight sums."""
		row = self.W.getrow(u)
		absolute_row = self.Wabs.getrow(u)
		numerator = (row @ self.E).toarray().ravel()
		denominator = (absolute_row @ self.B).toarray().ravel()
		return numerator, denominator

	def _residual_scores(self, u):
		numerator, denominator = self._num_den(u)
		denominator = denominator + self.damp
		return np.divide(
			numerator,
			denominator,
			out=np.zeros(self.n_items, dtype=np.float32),
			where=denominator != 0,
		)

	def score_user(self, u):
		"""Score all items using neighboring users' residuals or liked items."""
		user = int(u)
		if self.mode == "implicit":
			return (self.W.getrow(user) @ self.Bin).toarray().ravel()
		return self._residual_scores(user)

	def predict(self, u, i):
		"""Predict bias baseline plus residual user-neighborhood correction."""
		users = np.asarray(u, dtype=int).ravel()
		items = np.asarray(i, dtype=int).ravel()
		residual = np.empty(len(users), dtype=np.float32)
		for user in np.unique(users):
			if user not in self._predict_cache:
				self._predict_cache[user] = self._residual_scores(int(user))
			residual[users == user] = self._predict_cache[user][items[users == user]]
		return (
			self.bias.mu
			+ self.bias.bu[users]
			+ self.bias.bi[items]
			+ residual
		)
