"""Biased matrix factorization trained with stochastic gradient descent."""

import numpy as np
from numba import njit

from .base import Recommender


@njit(cache=True)
def sgd_epoch(u, i, r, order, mu, bu, bi, P, Q, lr, reg):
	"""Update factors once in shuffled order and return mean squared error."""
	squared_error = 0.0
	for position in range(len(order)):
		t = order[position]
		a = u[t]
		b = i[t]
		prediction = mu + bu[a] + bi[b]
		for factor in range(P.shape[1]):
			prediction += P[a, factor] * Q[b, factor]
		error = r[t] - prediction
		squared_error += error * error

		bu[a] += lr * (error - reg * bu[a])
		bi[b] += lr * (error - reg * bi[b])
		for factor in range(P.shape[1]):
			pa = P[a, factor]
			qb = Q[b, factor]
			P[a, factor] += lr * (error * qb - reg * pa)
			Q[b, factor] += lr * (error * pa - reg * qb)

	if len(order) == 0:
		return np.nan
	return squared_error / len(order)


class BiasedMF(Recommender):
	"""Global-mean, user/item-bias, and latent-factor rating model."""

	name = "mf"

	def __init__(
		self,
		n_factors=64,
		lr=0.005,
		reg=0.05,
		epochs=40,
		init_std=0.05,
		seed=42,
		patience=4,
		verbose=True,
	):
		self.n_factors = n_factors
		self.lr = lr
		self.reg = reg
		self.epochs = epochs
		self.init_std = init_std
		self.seed = seed
		self.patience = patience
		self.verbose = verbose

	def fit(self, train, n_users, n_items, val=None):
		"""Fit factors on train, using validation RMSE only for early stopping."""
		users = train["u"].to_numpy(dtype=np.int64)
		items = train["i"].to_numpy(dtype=np.int64)
		ratings = train["rating"].to_numpy(dtype=np.float64)
		if len(ratings) == 0:
			raise ValueError("train must contain at least one rating")

		self.n_users = n_users
		self.n_items = n_items
		self.mu = float(ratings.mean())
		self.bu = np.zeros(n_users, dtype=np.float64)
		self.bi = np.zeros(n_items, dtype=np.float64)
		rng = np.random.default_rng(self.seed)
		self.P = rng.normal(0.0, self.init_std, (n_users, self.n_factors)).astype(
			np.float64
		)
		self.Q = rng.normal(0.0, self.init_std, (n_items, self.n_factors)).astype(
			np.float64
		)

		if val is not None:
			val_users = val["u"].to_numpy(dtype=np.int64)
			val_items = val["i"].to_numpy(dtype=np.int64)
			val_ratings = val["rating"].to_numpy(dtype=np.float64)

		self.history = []
		best_val_rmse = np.inf
		best_parameters = None
		epochs_without_improvement = 0

		for epoch in range(1, self.epochs + 1):
			order = rng.permutation(len(ratings)).astype(np.int64)
			train_mse = sgd_epoch(
				users,
				items,
				ratings,
				order,
				self.mu,
				self.bu,
				self.bi,
				self.P,
				self.Q,
				self.lr,
				self.reg,
			)
			train_rmse = float(np.sqrt(train_mse))
			if not np.isfinite(train_rmse):
				raise ValueError(
					"training RMSE became NaN or infinite at epoch "
					f"{epoch} with learning rate {self.lr}"
				)

			epoch_result = {"epoch": epoch, "train_rmse": train_rmse}
			if val is not None:
				val_predictions = np.clip(
					self.predict(val_users, val_items), 0.5, 5.0
				)
				val_rmse = float(np.sqrt(np.mean((val_ratings - val_predictions) ** 2)))
				if not np.isfinite(val_rmse):
					raise ValueError(
						"validation RMSE became NaN or infinite at epoch "
						f"{epoch} with learning rate {self.lr}"
					)
				epoch_result["val_rmse"] = val_rmse
				if val_rmse < best_val_rmse - 1e-4:
					best_val_rmse = val_rmse
					self.best_epoch = epoch
					best_parameters = (
						self.bu.copy(),
						self.bi.copy(),
						self.P.copy(),
						self.Q.copy(),
					)
					epochs_without_improvement = 0
				else:
					epochs_without_improvement += 1
			self.history.append(epoch_result)

			if self.verbose:
				if val is None:
					print(f"epoch={epoch:03d} train_rmse={train_rmse:.4f}")
				else:
					print(
						f"epoch={epoch:03d} train_rmse={train_rmse:.4f} "
						f"val_rmse={epoch_result['val_rmse']:.4f}"
					)

			if (
				val is not None
				and self.patience is not None
				and epochs_without_improvement >= self.patience
			):
				break

		if val is not None:
			self.bu, self.bi, self.P, self.Q = best_parameters
		else:
			self.best_epoch = len(self.history)
		return self

	def predict(self, u, i):
		"""Predict ratings for arrays of user and item indices."""
		users = np.asarray(u, dtype=np.int64).ravel()
		items = np.asarray(i, dtype=np.int64).ravel()
		interaction = np.einsum("ij,ij->i", self.P[users], self.Q[items])
		return self.mu + self.bu[users] + self.bi[items] + interaction

	def score_user(self, u):
		"""Return item bias plus latent-factor affinity for one user."""
		return self.bi + self.Q @ self.P[int(u)]
