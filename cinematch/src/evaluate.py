"""Evaluation helpers for rating predictions and ranked recommendations."""

import numpy as np

from .config import RATING_MAX, RATING_MIN, REL_THRESHOLD, TOP_K
from .metrics import coverage, mae, novelty, ranking_metrics, rmse


def evaluate_rating(model, eval_df):
	"""Evaluate rating predictions after clipping them to the allowed range."""
	users = eval_df["u"].to_numpy()
	items = eval_df["i"].to_numpy()
	predictions = np.asarray(model.predict(users, items), dtype=float)
	predictions = np.clip(predictions, RATING_MIN, RATING_MAX)
	actual = eval_df["rating"].to_numpy(dtype=float)
	return {"rmse": rmse(actual, predictions), "mae": mae(actual, predictions)}


def evaluate_ranking(
	model,
	seen_df,
	eval_df,
	n_items,
	k=TOP_K,
	thr=REL_THRESHOLD,
	return_per_user=False,
):
	"""Evaluate leave-seen-out top-k ranking on users with relevant eval items."""
	if k <= 0:
		raise ValueError("k must be a positive integer")

	if len(seen_df):
		seen_by_user = seen_df.groupby("u")["i"].agg(set).to_dict()
		item_counts = np.bincount(seen_df["i"].to_numpy(), minlength=n_items)
		item_pop_share = item_counts / len(seen_df)
	else:
		seen_by_user = {}
		item_pop_share = np.zeros(n_items, dtype=float)

	relevant_by_user = (
		eval_df.loc[eval_df["rating"] >= thr]
		.groupby("u")["i"]
		.agg(set)
	)

	all_recs = []
	precisions = []
	recalls = []
	ndcgs = []
	average_precisions = []

	for user, relevant_items in relevant_by_user.items():
		scores = np.asarray(model.score_user(int(user)), dtype=float).copy()
		seen_items = list(seen_by_user.get(user, ()))
		if seen_items:
			scores[seen_items] = -np.inf

		candidates = np.flatnonzero(np.isfinite(scores))
		n_recs = min(k, len(candidates))
		if n_recs:
			candidate_scores = scores[candidates]
			selected = np.argpartition(candidate_scores, -n_recs)[-n_recs:]
			recommended = candidates[selected]
			order = np.lexsort((recommended, -scores[recommended]))
			recs = recommended[order].tolist()
		else:
			recs = []

		precision, recall, ndcg, average_precision = ranking_metrics(
			recs, relevant_items, k
		)
		all_recs.append(recs)
		precisions.append(precision)
		recalls.append(recall)
		ndcgs.append(ndcg)
		average_precisions.append(average_precision)

	users_evaluated = len(ndcgs)
	result = {
		f"precision@{k}": float(np.mean(precisions)) if users_evaluated else 0.0,
		f"recall@{k}": float(np.mean(recalls)) if users_evaluated else 0.0,
		f"ndcg@{k}": float(np.mean(ndcgs)) if users_evaluated else 0.0,
		f"map@{k}": float(np.mean(average_precisions)) if users_evaluated else 0.0,
		"coverage": coverage(all_recs, n_items),
		"novelty": novelty(all_recs, item_pop_share),
		"users_evaluated": users_evaluated,
	}
	if return_per_user:
		result["per_user_ndcg"] = ndcgs
	return result
