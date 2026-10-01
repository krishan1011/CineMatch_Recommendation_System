# src/models/base.py
import numpy as np


class Recommender:
    """Interface shared by movie recommendation models."""

    name = "base"

    def fit(self, train_df, n_users, n_items):
        """Fit the recommender from ratings and user/item dimensions."""
        raise NotImplementedError

    def predict(self, u, i):
        """Predict ratings for equal-length arrays of user and item indices."""
        raise NotImplementedError

    def score_user(self, u):
        """Return one recommendation score per item for user ``u``."""
        raise NotImplementedError
