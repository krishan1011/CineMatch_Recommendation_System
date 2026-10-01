# CineMatch Final Evaluation Findings

## Protocol and outcomes
All final models were refit on train + validation using the saved Phase 5-8 configurations. Test labels were not used for tuning. The best test RMSE model is hybrid_stack (RMSE 0.8767); the best test ranking model is item_knn_implicit (NDCG@10 0.1231). They differ because rating error optimizes numeric closeness, whereas NDCG rewards ordering relevant items near the top.

The hybrid scored NDCG@10 0.1196; the best single model, item_knn_implicit, scored 0.1231. The hybrid-minus-best-single difference is -0.0036. Hybrid-minus-MF-NDCG 95% CI [0.0711, 0.0945] and hybrid-minus-count-popularity CI [0.0327, 0.0564] both exclude zero. A paired CI against item_knn_implicit was not part of the requested bootstrap comparisons; the direct hybrid-minus-best-single point difference is reported without claiming significance. Hybrid is below the best single point estimate, so it did not add ranking lift over the strongest standalone test model. Its paired gains over MF-NDCG and count popularity do not establish a gain over that stronger standalone comparator.

The frozen Ridge stack achieved test RMSE 0.8767 and MAE 0.6746. Its fitted validation features use bias, item-kNN, user-kNN, and MF-RMSE predictions, with the saved Phase 8 coefficients applied to test predictions.

## Failure modes and mitigation
There are 1,702 test ratings for items with no trainval ratings. Bias RMSE is 1.0238 for these items versus 0.8927 for seen items; MF RMSE is 1.0004 versus 0.8682. Cold items have no collaborative history, so metadata/content and popularity remain useful fallbacks. For the lightest measured user bucket (20-50 trainval ratings), RMSEs were bias 0.947, hybrid_stack 0.937, item_knn 0.935, mf 0.939, user_knn 0.936; personalized evidence remains sparse there. Highest observed genre RMSEs were Horror (MF RMSE 0.973), Sci-Fi (MF RMSE 0.943), Children (MF RMSE 0.936). Genre-level differences and low-support tails should be treated cautiously.

The count-popularity baseline reaches test NDCG@10 0.0751 but coverage is only 0.0101; this shows why accuracy alone misses catalog concentration. The error analysis also reports recommendation-popularity deciles, intra-list content diversity, calibration, and the largest MF errors. The 20 worst MF errors include Burnt by the Sun (Utomlyonnye solntsem) (1994), Fast Five (Fast and the Furious 5, The) (2011), Sling Blade (1996), Zoolander (2001), Dallas Buyers Club (2013); errors are mostly in one direction. These are explicit preference disagreements in observed ratings, not proof of corrupt records.

## Limitations and next steps
This system uses explicit ratings from one MovieLens dataset, tied timestamps make ordering within some temporal split boundaries arbitrary, and all current metrics are offline. Bootstrap intervals resample users, not items or time periods. Next steps are a prospective user study, stronger item metadata/text, better cold-item onboarding, and time-aware evaluation on additional datasets.
