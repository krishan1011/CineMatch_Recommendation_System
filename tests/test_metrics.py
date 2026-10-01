import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, evaluate_ranking
from src.metrics import ranking_metrics, rmse


def test_rmse_zero_case():
	assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_perfect_ranking_metrics():
	precision, recall, ndcg, average_precision = ranking_metrics(
		[1, 2, 3], {1, 2, 3}, 3
	)

	assert (precision, recall, ndcg, average_precision) == (1.0, 1.0, 1.0, 1.0)


def test_ranking_metrics_without_hits_are_zero():
	assert ranking_metrics([0, 1], {2}, 2) == (0.0, 0.0, 0.0, 0.0)


def test_ndcg_prefers_an_early_relevant_item():
	early = ranking_metrics([0, 1], {0}, 2)[2]
	late = ranking_metrics([1, 0], {0}, 2)[2]

	assert early > late


def test_recall_uses_all_relevant_items_in_denominator():
	recall = ranking_metrics([1, 2, 3], {1, 2, 4}, 3)[1]

	assert recall == 2 / 3


def test_evaluate_ranking_masks_seen_items():
	class FixedScores:
		def score_user(self, user):
			return np.array([10.0, 9.0, 8.0, 7.0])

	seen = pd.DataFrame({"u": [0], "i": [0], "rating": [4.0]})
	evaluation = pd.DataFrame({"u": [0], "i": [1], "rating": [5.0]})
	result = evaluate_ranking(
		FixedScores(), seen, evaluation, n_items=4, k=1,
		return_per_user=True, return_recs=True,
	)

	assert result["users_evaluated"] == 1
	assert result["per_user_ndcg"] == {0: 1.0}
	assert result["per_user_recs"] == {0: [1]}


def test_bootstrap_ci_constant_input_has_exact_interval():
	mean, lower, upper = bootstrap_ci({user: 0.4 for user in range(20)}, n_boot=200)

	assert np.isclose(mean, 0.4)
	assert lower == upper == 0.4


def test_bootstrap_ci_contains_sample_mean():
	mean, lower, upper = bootstrap_ci({0: 0.1, 1: 0.5, 2: 0.8, 3: 1.0}, n_boot=1000, seed=3)

	assert lower <= mean <= upper
