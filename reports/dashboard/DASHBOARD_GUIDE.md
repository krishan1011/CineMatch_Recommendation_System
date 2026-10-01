# CineMatch Power BI Dashboard Guide

## 1. Load the BI exports
1. In Power BI Desktop choose **Home > Get Data > More... > Folder** and browse to `cinematch/reports/bi`.
2. Choose **Transform Data**, not **Combine**. These CSVs have different schemas, so they must become separate queries rather than one combined table.
3. In Power Query, duplicate the folder query once per file. In each copy filter `Name` to the required filename, keep its `Content`, open that binary with **Combine Files / CSV**, and promote the first row to headers. Load these tables: `bi_ratings_summary`, `bi_movies`, `bi_users`, `bi_sorted_counts`, `bi_kpis`, `results_test`, `results_val`, `bi_model_labels`, `bi_error_by_bucket`, `bi_cold_start`, `bi_recs_sample`, `bi_popularity_bias`, and `bi_diversity`.
4. To refresh on another computer, edit the Folder `Source` step (or define a text parameter such as `BiFolderPath` and use it in that step), then choose **Home > Refresh**. Keep the exported filenames unchanged.

## 2. Model the data
Use single-direction, one-to-many relationships from dimensions to facts:
- `bi_model_labels[model]` (one) to `results_test[model]` (many).
- `bi_model_labels[model]` to `bi_error_by_bucket[model]`, `bi_cold_start[model]`, `bi_popularity_bias[model]`, `bi_diversity[model]`, and `bi_recs_sample[model]`.
- `bi_users[user_id]` (one) to `bi_recs_sample[user_id]` (many).
- Avoid relating `bi_movies[movie_id]` directly to recommendations: `bi_movies` intentionally has one row per movie-genre, so `movie_id` repeats. The recommendation table already carries title and genres.

For error charts, create a calculated display column so duplicate bucket names in different bucket types sort unambiguously:
```DAX
bucket_display = bi_error_by_bucket[bucket_type] & " | " & bi_error_by_bucket[bucket]
```
Select `bi_error_by_bucket[bucket_display]`, then **Column tools > Sort by column > bucket_order**. Keep `n_ratings`, `rank`, `popularity_decile`, and `user_id` as numeric columns; set date fields to Date and `is_hybrid` to True/False.

## 3. Measures
Create these measures (values in `results_test` are long-form, so each metric measure filters both split and metric):
```DAX
Avg NDCG@10 =
CALCULATE(
    AVERAGE(results_test[value]),
    results_test[split] = "test",
    results_test[metric] = "ndcg@10"
)

Best Model NDCG (test split) =
MAXX(
    ALL(bi_model_labels[model]),
    CALCULATE([Avg NDCG@10])
)

Lift vs Popularity =
[Avg NDCG@10]
    - CALCULATE(
        [Avg NDCG@10],
        REMOVEFILTERS(bi_model_labels[model]),
        bi_model_labels[model] = "popularity_count"
    )

Sparsity % = MAX(bi_kpis[sparsity])

Avg RMSE =
CALCULATE(
    AVERAGE(results_test[value]),
    results_test[split] = "test",
    results_test[metric] = "rmse"
)

Coverage =
CALCULATE(
    AVERAGE(results_test[value]),
    results_test[split] = "test",
    results_test[metric] = "coverage"
)

NDCG Matrix Color =
IF([Avg NDCG@10] = [Best Model NDCG (test split)], "#147D78", "#D9DEE1")
```
Format `Sparsity %` as a percentage. Use `NDCG Matrix Color` for **Conditional formatting > Background color > Format style: Field value**. Highlight only the best model in teal; keep other models grey.

## 4. Build six pages

### Page 1: Data Overview
- Title: **“A sparse catalog with a concentrated rating head”**.
- Four Card visuals: `bi_kpis[n_users]`, `[n_movies]`, `[n_ratings]`, and `[Sparsity %]`.
- Line chart: `bi_ratings_summary[key]` on X and `count` on Y; visual filter `summary_type = rating_value`, sorted by the numeric rating key.
- Bar chart: `bi_ratings_summary[key]` by `count`, visual filter `summary_type = genre`, descending count.
- Add slicers for `bi_ratings_summary[summary_type]` and `bi_ratings_summary[key]`.

### Page 2: Long Tail
- Title: **“The top 10% of movies hold 60.10% of ratings”**.
- Line chart using `bi_sorted_counts[rank]` on X and `cumulative_share` on Y; set X to continuous and sort ascending by rank. Add `rating_count` to tooltip.
- Scatter chart from `bi_movies`: `rating_count` on X, `mean_rating` on Y, `popularity_decile` as legend, `title` as tooltip; optionally filter to one `genre` with a slicer.
- Add `bi_movies[genre]` and `bi_movies[popularity_decile]` slicers.

### Page 3: Model Comparison
- Title: **“Implicit Item-kNN leads test ranking; the stack leads rating error”**.
- Matrix: rows `bi_model_labels[label]`; values `[Avg RMSE]`, `[Avg NDCG@10]`, `[Coverage]`, plus `Avg MAE` and `Avg MAP@10` measures filtered to their matching `results_test[metric]` values. Apply `NDCG Matrix Color` to the NDCG cell background.
- Clustered bar chart: axis `bi_model_labels[label]`, value `[Avg NDCG@10]`, descending; turn on data labels.
- Card: `[Best Model NDCG (test split)]`; add slicer `bi_model_labels[is_hybrid]`.
- `Avg MAE` and `Avg MAP@10` follow the same pattern as the other metric measures, using `results_test[metric] = "mae"` and `"map@10"` respectively.

### Page 4: Error Analysis
- Title: **“Errors rise for sparse users and unseen items”**.
- Line chart for `bucket_type = user_activity`: axis `bucket_display`, legend `bi_error_by_bucket[model]`, value `rmse`; a second line chart uses `ndcg`. Sort bucket display by `bucket_order`.
- Bar chart for `bucket_type = item_popularity`: axis `bucket_display`, legend model, value `rmse`; data labels on.
- Bar chart for `bucket_type = genre`: axis `bucket`, legend model, value `rmse`; sort genre ascending.
- Slicers: `bucket_type`, `model` label. Rating-only and ranking-only cells are blank by design.

### Page 5: Cold Start
- Title: **“The tuned cold profile improves after three ratings”**.
- Line chart: X `bi_cold_start[n_ratings]`, Y `bi_cold_start[ndcg_at_10]`, legend `bi_model_labels[label]`; sort X ascending and enable data labels.
- Slicer: `bi_cold_start[model]` through the model-label relationship. Emphasize `hybrid` versus `hybrid_cold` around 3, 5, and 10 ratings.

### Page 6: Recommendation Inspector
- Title: **“Inspect what the engine recommends for each user”**.
- Slicer: `bi_users[user_id]` (single select).
- Table: `bi_recs_sample[rank]`, `[title]`, `[genres]`, `[model]`; filter rank to 1-10 and sort ascending.
- Bar chart: `bi_popularity_bias[popularity_decile]` by `[recommendation_share]`, legend model; add `[test_relevant_share]` as a line or tooltip.
- Table or cards: `bi_diversity[model]`, `[mean_intra_list_diversity]`, `[users_evaluated]`.
- Add a model-label slicer so a user can compare popularity, content, MF, item-kNN, and hybrid lists.

Use one accent color (teal `#147D78`) for the best model and grey `#D9DEE1` for others. Keep the palette to at most six colors, use sorted bars with data labels, and avoid 3D visuals.

## 5. Validation checklist
Use `reports/results_test.csv` as the source of truth. These values should match the Model Comparison table (rounding to four decimals):

| Model | RMSE | MAE | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|
| bias | 0.9044 | 0.6986 | 0.0555 | 0.0053 |
| content | blank | blank | 0.0694 | 0.0563 |
| hybrid | blank | blank | 0.1196 | 0.0482 |
| hybrid_stack | 0.8767 | 0.6746 | blank | blank |
| item_knn | 0.8811 | 0.6737 | 0.1049 | 0.0886 |
| item_knn_implicit | blank | blank | 0.1231 | 0.0550 |
| mf | 0.8799 | 0.6754 | 0.0370 | 0.0284 |
| mf_ndcg | blank | blank | 0.0360 | 0.0251 |
| popularity_bayes | blank | blank | 0.0625 | 0.0059 |
| popularity_count | blank | blank | 0.0751 | 0.0101 |
| random | blank | blank | 0.0017 | 0.4519 |
| user_knn | 0.8866 | 0.6797 | 0.0910 | 0.0813 |
| user_knn_implicit | blank | blank | 0.1070 | 0.0216 |

The best rating RMSE is hybrid_stack at **0.8767**; the best ranking NDCG@10 is item_knn_implicit at **0.1231**. The overview KPI check is 610 users, 9,742 movies, 100,836 ratings, and 98.3032% sparsity.

## 6. Screenshots and handoff
In Power BI Desktop select each page, use **Win+Shift+S** to capture the report canvas, and save `page-1-overview.png`, `page-3-model-comparison.png`, and `page-5-cold-start.png` under `reports/dashboard/`. Save the finished `.pbix` under `reports/dashboard/` and commit it only if its file size is below 25 MB; otherwise keep the PBIX local and commit the guide and screenshots only.
