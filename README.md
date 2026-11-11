Domain: Support

# Swiggy Support Decision Intelligence

This project answers three support-analytics questions on a synthetic set of 800 tickets: whether the 8-minute resolution-time planning assumption holds, which tickets will escalate to a senior team, and what ticket groups the queue contains. It also sizes two neural networks a later project could build. Everything is computed in plain Python with NumPy; every number below is read from the files in `outputs/`.

The data is synthetic and comes from `generate_data.py`. It does not reflect Swiggy's real records, products, policies or performance.

## How to run

From a clean clone, at the repository root (Python 3.10 or newer):

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` prints `800 10 170 7cb48ef89914049d76954530b6482e04` and writes `data/records.csv`. Each Part script rewrites its own files in `outputs/`. `common.py` holds the code shared by the Parts (data loading, the stratified split, logistic regression, k-nearest neighbours, metrics and file writers); importing it does not run any Part.

| File | Produced by |
|---|---|
| `outputs/part1_results.txt` | `part1.py` |
| `outputs/part2_results.txt`, `part2_thresholds.csv`, `part2_knn_cv.csv` | `part2.py` |
| `outputs/part3_results.txt`, `part3_segments.csv`, `part3_hierarchical.csv` | `part3.py` |
| `outputs/part4_results.txt`, `part4_splits.csv`, `part4_voting.csv`, `part4_adaboost.csv`, `part4_architecture.csv` | `part4.py` |

## Part 1 findings

**R1.1 — The 8-minute assumption.** H0 (μ = 8.0 minutes) is rejected: `z_stat` = 2.8930, which is above the critical value 1.96 for a two-tailed test at α = 0.05. The 95% interval for the mean resolution time is 8.3026 to 9.5738 minutes, and it does not contain 8.0. A Type I error here would mean concluding that the average resolution time differs from 8 minutes when it really is 8 minutes, so the team would re-plan staffing it did not need to change. The gap is `mean_value` − 8.0 = 8.9382 − 8.0 = 0.9382 minutes, close to a minute per ticket, or about 12% above the plan. I think this matters in practice: staffing scales linearly with handling time, so a plan built on 8 minutes understaffs the queue by roughly that share. The distribution is also very skewed (`sd_value` = 9.1723, about as large as the mean), so a single average hides a long tail of slow tickets.

**R1.2 — Sample size.** Estimating the mean to within ±0.5 minutes at 95% confidence needs `n_required` = 1293 tickets, and only 800 are available. With 800 rows the half-width of the interval is 9.5738 − 8.9382 = 0.6356 minutes, wider than the half minute the team wants. The planning team should collect about 500 more tickets (or accept a ±0.64-minute margin) before treating the average as precise to half a minute.

**R1.3 — Gradient descent vs closed form.** `gd_slope_gap` = 0.0000 (below 0.001): after 500 updates gradient descent gave the same line as the closed form (`gd_slope` = `reg_slope` = 0.6701, `gd_intercept` = `reg_intercept` = −3.6598). The mean-squared-error cost is a convex bowl with a single minimum, the least-squares solution, and on the standardised predictor a learning rate of 0.1 steps steadily down to it. The cost was already flat at 40.3698 by update 100 and stayed there at update 500.

**R1.4 — Funnel check.** The raw model shows a clear funnel. `resid_sd_ratio_raw` = 3.7525: the residual SD is 8.6891 minutes for the half of tickets with the highest fitted values and 2.3155 for the lowest half, so errors grow with predicted resolution time. The log-log model removes the funnel: `resid_sd_ratio_log` = 1.0396 (0.5369 vs 0.5165), so the spread is almost the same in both halves.

**R1.5 — Multiple regression.** Excluding the constant, the coefficients with an absolute value above 0.05 are ln(X4) `account_value_inr_thousands` (1.1576), X2 `earlier_ticket_count_90d` (0.0940), X3 `sla_consumed_ratio` (0.1067) and X6 `reopened_flag` (0.1518). X1, X5 and X7 fall below the threshold (−0.0074, 0.0028, 0.0172). As the brief says, this is a mechanical cut-off on coefficients measured in different units, not a ranking of importance. The extra predictors earn their place: `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561 for ln(X4) alone. Adjusted R² penalises each added predictor, and it still rises by about 0.08.

## Part 2 findings

**R2.1 — Learning rate.** After 200 updates the training losses are, from lowest to highest: η = 1.0 (0.3703), η = 0.1 (0.3776), η = 0.01 (0.5091). A learning rate that is too small moves so slowly that it is still far from the minimum after 200 updates, while a large one (1.0) is stable on this standardised, convex problem and reaches the minimum fastest (it is already at the 1000-update loss of 0.3704).

**R2.2 — Strongest predictors.** The three largest absolute weights are X3 `sla_consumed_ratio` (0.9013), X1 `customer_tenure_months` (−0.5554) and X2 `earlier_ticket_count_90d` (0.5505). The largest odds ratio is `odds_ratio_X3` = 2.4629: a ticket that reaches an agent with one training SD more of its resolution-time target already used has about 2.5 times the odds of escalating, other factors held fixed.

**R2.3 — Why accuracy misleads.** `blind_accuracy` = 0.7875 for a "model" that never flags anything, against a τ = 0.5 accuracy of 0.8042. A model that catches no escalations scores only 0.017 below the real model, because 79% of test tickets do not escalate. Accuracy rewards the majority class and says little about catching escalations. Lowering τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (24 → 38 of 51 escalations caught) and lowered precision from 0.5455 to 0.4634 (false alarms rose from 20 to 44).

**R2.4 — Logistic vs KNN.** Logistic regression at τ = 0.3 has `f1` 0.5714 and `recall` 0.7451. KNN (`knn_best_k` = 5) has `f1` 0.3696 and `recall` 0.3333. The logistic model is clearly better on both: KNN misses two-thirds of escalations.

**R2.5 — What to deploy.** Deploy logistic regression at τ = `cost_tau` = 0.15. On the test rows it costs `cost_test_at_tau` = INR 6,050, against INR 11,800 at τ = 0.5 (`cost_test_at_0.5`) and INR 20,400 for flagging nobody (`cost_test_flag_nobody`). Because a missed escalation (INR 400) costs eight times a false alarm (INR 50), the cost-optimal threshold is low. The threshold was chosen on training rows so that the test rows stay an unseen check: choosing it on the test rows would tune it to those 240 tickets and make the test cost an optimistic, biased estimate.

## Part 3 findings

**R3.1 — Elbow.** The four successive WCSS drops are 659.8282 (K 2→3), 108.0739 (3→4), 171.0575 (4→5) and 45.5466 (5→6). The elbow supports K = 3 (`elbow_k` = 3): moving from 3 to 4 clusters gains less than half of what moving from 2 to 3 gained. The drops do not shrink steadily (3→4 is smaller than 4→5). This is because the pinned starting rows give one reproducible solution, not necessarily the global optimum for K = 4, so the K = 4 WCSS is higher than the best achievable.

**R3.2 — Segments (K = 3, from `part3_segments.csv`).**

| Segment | Size | Behaviour | Business name | Action |
|---|---|---|---|---|
| A | 322 | Ticket reaches an agent early (SLA used 0.225), few earlier tickets (1.95), satisfaction 62.2, resolved in 3.60 min, escalation rate 0.0186 | Quick early-caught tickets | Steer to self-service and fast-lane agents; no supervisor review needed |
| B | 255 | Most repeat contact (6.07 tickets in 90 days), highest spend (INR 28.7 thousand), slowest to resolve (17.12 min), escalation rate 0.2353 | Repeat-contact, high-value, slow tickets | Route to experienced agents with the customer's ticket history open, and staff this queue on about 17 minutes per ticket, not 8 |
| C | 223 | Arrives late (SLA used 0.712), lowest satisfaction (38.5), escalation rate 0.4664, resolved in 7.29 min | Late-arriving, at-risk tickets | Put in a priority queue at intake and send straight to the supervisor-review alert |

Segment C has the highest `outcome_rate`, 0.4664: almost half of its tickets escalate, against 0.0186 in segment A.

**R3.3 — Hierarchical clustering.** Single linkage, cut in its largest gap (between merges at 1.4 and 2.8), gives 4 clusters: {E1}, {E2, E3, E6}, {E4}, {E5}. Complete linkage, cut in its largest gap (between 2.9 and 7.9), gives 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Single linkage joins two groups through their closest pair, so the two cohort groups are bridged cheaply by E5–E6 (3.4) and the final merge sits close to the earlier ones, leaving the largest gap low in the tree. Complete linkage uses the farthest pair, so the final merge costs the full E1–E2 distance (7.9) and the largest gap sits at the top, separating two clear groups.

**R3.4 — PCA.** The mean vector of the standardised matrix is all zeros (each column was standardised over all 800 rows), so centring changes nothing. PC1 contrasts long tenure (X1, +0.5317) and high satisfaction (X5, +0.4418) against a high SLA-consumed ratio (X3, −0.5187), many earlier tickets (X2, −0.4100) and, more weakly, higher spend (X4, −0.2916). A high PC1 score means a calm, established customer; a low score means friction and a ticket already late. PC1 explains 52.27% of the variance, and `components_for_80pct` = 3 components are needed to keep at least 80% (cumulative 0.8616). The segment means on PC1 are A +1.7773, B −0.7831 and C −1.6709, so PC1 orders the segments the same way as their escalation rates (0.0186, 0.2353, 0.4664). The first component on its own works as a low-risk to at-risk axis.

## Part 4 findings

**R4.1 — Best split.** S4 (X1 ≤ 24, tenure of two years or less) is best by both information gain (0.1252) and Gini gain (0.0606). It separates the classes most cleanly: 86 of 188 tickets (45.7%) on the "yes" side escalate, against 33 of 372 (8.9%) on the "no" side, starting from a root entropy of 0.7462. A tree grown until every leaf is pure keeps splitting until it isolates single tickets, so it memorises noise in the 560 training rows and generalises poorly to new tickets. A maximum depth (or a minimum number of records per leaf) stops it from growing too deep.

**R4.2 — Which model to ship.** I would ship logistic regression: at τ = 0.3 it has the best `f1` (0.5714) with `recall` 0.7451. Combining models did not help: `hard_vote` reached `f1` 0.5344 and `recall` 0.6863, and `soft_vote` matched logistic recall (0.7451) with a lower `f1` (0.5507) because of five extra false alarms. The stump catches one more escalation (`recall` 0.7647) but has a lower `f1` (0.5532), and KNN is weak (`f1` 0.3696, `recall` 0.3333), which pulls the hard vote down.

**R4.3 — AdaBoost.** After round 1 normalisation the two points misclassified by h1 (points 3 and 7) carry a total weight of 0.2500 + 0.2500 = 0.5000, exactly half of the total. That is the effect of AdaBoost's update: h1 now looks like a coin flip on the reweighted data. Round 2 trusts h2 more (`round2_alpha` = 0.8047 vs `round1_alpha` = 0.5493) because h2 gets the heavily weighted points 3 and 7 right and only misses points 1 and 5, so its weighted error is 0.1667, lower than h1's 0.2500.

**R4.4 — Network sizing.**

Model A, option 2 (convolutional network on a 28 × 28 × 1 input):

| Layer | Output shape (H × W × D) | Parameters |
|---|---|---|
| Input | 28 × 28 × 1 | 0 |
| Conv 5×5, 6 filters, stride 1, padding 2 | 28 × 28 × 6 | 156 |
| ReLU | 28 × 28 × 6 | 0 |
| Max-pool 2×2, stride 2 | 14 × 14 × 6 | 0 |
| Conv 5×5, 16 filters, stride 1, padding 0 | 10 × 10 × 16 | 2,416 |
| ReLU | 10 × 10 × 16 | 0 |
| Max-pool 2×2, stride 2 | 5 × 5 × 16 | 0 |
| Flatten | 1 × 1 × 400 | 0 |
| Dense 32 + ReLU | 1 × 1 × 32 | 12,832 |
| Dense 10 (output) | 1 × 1 × 10 | 330 |
| **Total** | | **15,734** |

The fully connected option 1 has 13,002 parameters (`mlp_total`). The convolutional network has a few more (`cnn_total` = 15,734) but reuses each small filter across the whole image. The gap shows at larger inputs: a dense layer from a 128 × 128 × 3 input to 1,024 neurons needs 50,332,672 parameters, while a 3 × 3 convolution with 64 filters needs 1,792.

Model B (recurrent network): W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128, with b_h of length 128 and b_y of length 16, for `rnn_total` = 26,768. The total does not grow with longer sequences because the same three matrices and two biases are reused at every time step; a longer ticket only means more steps through the same weights.

For 10 classes I would use ReLU in the hidden layers and a softmax output trained with categorical cross-entropy loss. In a very deep plain network the gradient is multiplied by one layer's derivatives after another on the way back, so it can shrink towards zero (vanish) before it reaches the early layers, which then stop learning. A skip connection adds the layer's input directly to its output (y = F(x) + x), giving the gradient an identity path that bypasses the stacked layers, so the early layers keep receiving a usable signal.

## Model-choice memo

**To:** Head of Customer-Support Operations
**Subject:** Which escalation model to put into production

I recommend the logistic-regression model for flagging tickets for supervisor review before they escalate.

On the 240 held-out test tickets, at a threshold of 0.3, it caught 38 of 51 escalations (recall 0.7451) with the best F1 of the five models compared (0.5714). Combining models did not beat it: the hard vote scored F1 0.5344 and the soft vote 0.5507. The nearest-neighbour model missed two-thirds of escalations (recall 0.3333).

For production I recommend a lower threshold of 0.15. It was chosen on the training tickets using your cost figures (INR 50 per unnecessary review, INR 400 per missed escalation). On the test tickets it costs INR 6,050, against INR 11,800 at the default 0.5 threshold and INR 20,400 if we flag nobody. A missed escalation costs eight times a false alarm, so it pays to review generously.

The model's main signals are easy to act on: tickets that reach an agent with much of their resolution-time target already used (each extra standard deviation multiplies the odds of escalation by 2.46), newer customers and customers with many recent tickets.

The main limitation is the evidence base: one fixed split with only 51 escalations in the test set, on synthetic data. A few tickets more or fewer would move the cost figures noticeably, so the threshold should be re-checked on live tickets during a monitored pilot before full rollout.

Neither neural network (Model A for order photos, Model B for ticket text) is worth building before this structured model is in production, because the structured model already captures much of the escalation risk at low cost, and the text and image models need labelled data we do not yet have.

## Limitations

1. **Synthetic data.** All 800 tickets come from a generator with known, smooth relationships (log-normal resolution times, a logistic escalation rule). Real support data will be messier (missing values, outliers, changes over time), so the numbers here show the method, not Swiggy's actual performance.
2. **One fixed split and few positives.** Every Part 2 and Part 4 test result comes from a single 560/240 split with only 51 escalations in the test set. One ticket changes recall by about 0.02 and the cost by INR 50 to 400, so the model comparisons and the chosen threshold (0.15) carry real uncertainty; the K picked by cross-validation (5) also holds only for these pinned folds.
3. **Fixed cost figures.** The INR 50 and INR 400 costs are approximate planning figures. If a missed escalation really costs much more or less, the cost-optimal threshold moves.
4. **Pinned k-means starting rows.** The segments come from one pinned start. As the uneven WCSS drops show, other starts can reach lower WCSS for some K, so the segment boundaries are reproducible but not necessarily optimal.
5. **Simple models only.** The regression and the classifier are linear in their inputs (after logs and standardisation) with no interactions, and the Part 1 interval relies on the normal approximation for a strongly skewed variable.
6. **No time dimension.** Tickets are treated as independent and exchangeable. A real deployment would need to test the model on later tickets than it was trained on.
