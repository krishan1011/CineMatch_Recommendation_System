"""Rating prediction and top-k recommendation metrics."""

import numpy as np


def rmse(y, p):
	"""Return root mean squared error between true and predicted ratings."""
	errors = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
	return float(np.sqrt(np.mean(errors**2)))


def mae(y, p):
	"""Return mean absolute error between true and predicted ratings."""
	errors = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
	return float(np.mean(np.abs(errors)))


def ranking_metrics(recs, relevant_set, k):
	"""Return precision, recall, NDCG, and average precision for one user."""
	if k <= 0:
		return 0.0, 0.0, 0.0, 0.0

	relevant = set(relevant_set)
	n_relevant = len(relevant)
	recommendations = list(recs)[:k]
	hits = []
	seen = set()
	for item in recommendations:
		hit = item in relevant and item not in seen
		hits.append(hit)
		seen.add(item)

	n_hits = sum(hits)
	precision = n_hits / k
	recall = n_hits / n_relevant if n_relevant else 0.0

	discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
	dcg = float(np.sum(np.asarray(hits, dtype=float) * discounts))
	ideal_hits = min(n_relevant, k)
	idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
	ndcg = dcg / idcg if idcg else 0.0

	denominator = min(n_relevant, k)
	if denominator:
		cumulative_precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
		average_precision = float(np.sum(cumulative_precision * hits) / denominator)
	else:
		average_precision = 0.0

	return float(precision), float(recall), float(ndcg), average_precision


def coverage(all_recs, n_items):
	"""Return the fraction of catalog items recommended at least once."""
	if n_items <= 0:
		return 0.0
	recommended = {item for recs in all_recs for item in recs}
	return float(len(recommended) / n_items)


def novelty(all_recs, item_pop_share):
	"""Return mean self-information of recommended items from popularity shares."""
	recommended = [item for recs in all_recs for item in recs]
	if not recommended:
		return 0.0
	shares = np.asarray([item_pop_share[item] for item in recommended], dtype=float)
	return float(np.mean(-np.log2(shares + 1e-12)))
