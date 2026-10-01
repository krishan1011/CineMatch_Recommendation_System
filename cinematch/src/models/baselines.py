"""Simple random, popularity, and regularized-bias recommenders."""

import numpy as np

from .base import Recommender


class RandomRec(Recommender):
	"""Random ranking baseline with train-mean rating predictions."""

	name = "random"

	def __init__(self, seed=42):
		self.seed = seed

	def fit(self, train_df, n_users, n_items):
		"""Store the train mean and initialize the seeded random generator."""
		self.mu = float(train_df["rating"].mean())
		self.n_users = n_users
		self.n_items = n_items
		self.rng = np.random.default_rng(self.seed)
		return self

	def predict(self, u, i):
		"""Predict the global mean rating for each user-item pair."""
		return np.full(np.asarray(u).shape, self.mu, dtype=float)

	def score_user(self, u):
		"""Return seeded random recommendation scores for the item catalog."""
		return self.rng.random(self.n_items)


class Popularity(Recommender):
	"""Rank items by raw count or a train-only Bayesian mean estimate."""

	def __init__(self, mode="bayes", m=15):
		if mode not in {"bayes", "count"}:
			raise ValueError("mode must be 'bayes' or 'count'")
		self.mode = mode
		self.m = m
		self.name = f"popularity_{mode}"

	def fit(self, train_df, n_users, n_items):
		"""Compute train-only item counts, means, and smoothed item scores."""
		users = train_df["u"].to_numpy()
		items = train_df["i"].to_numpy()
		ratings = train_df["rating"].to_numpy(dtype=float)
		self.n_users = n_users
		self.n_items = n_items
		self.mu = float(ratings.mean())
		self.counts = np.bincount(items, minlength=n_items)
		rating_sums = np.bincount(items, weights=ratings, minlength=n_items)
		self.item_means = np.divide(
			rating_sums,
			self.counts,
			out=np.full(n_items, self.mu, dtype=float),
			where=self.counts > 0,
		)
		self.bayes_scores = (rating_sums + self.m * self.mu) / (self.counts + self.m)
		self.scores = self.bayes_scores if self.mode == "bayes" else self.counts.astype(float)
		return self

	def predict(self, u, i):
		"""Predict Bayesian item means, or the global mean for count-only mode."""
		items = np.asarray(i, dtype=int)
		if self.mode == "bayes":
			return self.bayes_scores[items]
		return np.full(items.shape, self.mu, dtype=float)

	def score_user(self, u):
		"""Return train-only popularity scores for every item."""
		return self.scores.copy()


class BiasModel(Recommender):
	"""Global-mean model with regularized item and user biases."""

	name = "bias"

	def __init__(self, lam_i=10, lam_u=15):
		self.lam_i = lam_i
		self.lam_u = lam_u

	def fit(self, train_df, n_users, n_items):
		"""Fit global mean and regularized item-then-user biases on train data."""
		users = train_df["u"].to_numpy()
		items = train_df["i"].to_numpy()
		ratings = train_df["rating"].to_numpy(dtype=float)
		self.n_users = n_users
		self.n_items = n_items
		self.mu = float(ratings.mean())

		item_counts = np.bincount(items, minlength=n_items)
		item_residual_sums = np.bincount(
			items, weights=ratings - self.mu, minlength=n_items
		)
		self.bi = item_residual_sums / (self.lam_i + item_counts)

		user_counts = np.bincount(users, minlength=n_users)
		user_residual_sums = np.bincount(
			users,
			weights=ratings - self.mu - self.bi[items],
			minlength=n_users,
		)
		self.bu = user_residual_sums / (self.lam_u + user_counts)
		return self

	def predict(self, u, i):
		"""Predict ratings from the global mean and learned user/item biases."""
		users = np.asarray(u, dtype=int)
		items = np.asarray(i, dtype=int)
		return self.mu + self.bu[users] + self.bi[items]

	def score_user(self, u):
		"""Return item-bias scores for ranking items for user ``u``."""
		return self.bi.copy()
