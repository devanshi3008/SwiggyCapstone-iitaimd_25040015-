"""Part 2 - Classification and evaluation (560 training rows / 240 test rows)."""
import math

import numpy as np

from common import (prepare_split, train_logistic, fit_part2_logistic, sigmoid, bce,
                    confusion, metrics, knn_cv, knn_predict, best_k,
                    write_results, write_csv)


def money_cost(fp, fn):
    return 50 * fp + 400 * fn


def main():
    S = prepare_split()
    Xtr, ytr, Xte, yte = S["Xtr"], S["ytr"], S["Xte"], S["yte"]
    out = []

    # 2.1 split
    out += [("train_rows", int(ytr.size)), ("train_positives", int(ytr.sum())),
            ("test_rows", int(yte.size)), ("test_positives", int(yte.sum()))]

    # 2.2 logistic regression from scratch
    w, b, losses = fit_part2_logistic(S)
    for t in (0, 1, 10, 100, 1000):
        out.append((f"loss_epoch_{t}", losses[t]))
    for j in range(7):
        out.append((f"weight_X{j + 1}", float(w[j])))
    out.append(("bias", float(b)))
    p_te = sigmoid(Xte @ w + b)
    p_tr = sigmoid(Xtr @ w + b)
    out.append(("test_bce", bce(p_te, yte)))

    # 2.3 learning-rate effect
    for eta in (0.01, 0.1, 1.0):
        _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
        out.append((f"loss_epoch_200_eta_{eta}", l[200]))

    # 2.4 odds ratios per one training SD
    for j in range(7):
        out.append((f"odds_ratio_X{j + 1}", math.exp(w[j])))

    # 2.5 two thresholds on the test rows
    for tau in (0.5, 0.3):
        tp, fp, tn, fn = confusion(p_te >= tau, yte)
        m = metrics(tp, fp, tn, fn)
        out += [(f"tau_{tau}_tp", tp), (f"tau_{tau}_fp", fp),
                (f"tau_{tau}_tn", tn), (f"tau_{tau}_fn", fn),
                (f"tau_{tau}_accuracy", m["accuracy"]),
                (f"tau_{tau}_precision", m["precision"]),
                (f"tau_{tau}_recall", m["recall"]),
                (f"tau_{tau}_f1", m["f1"]),
                (f"tau_{tau}_specificity", m["specificity"])]
    tp, fp, tn, fn = confusion(np.zeros(yte.size), yte)
    out.append(("blind_accuracy", metrics(tp, fp, tn, fn)["accuracy"]))

    # 2.6 threshold sweep (ROC points)
    rows = []
    for k in range(1, 10):
        tau = k / 10
        tp, fp, tn, fn = confusion(p_te >= tau, yte)
        m = metrics(tp, fp, tn, fn)
        rows.append([f"{tau:.4f}", tp, fp, tn, fn, m["recall"], m["fpr"]])
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

    # 2.7 KNN with 5-fold cross-validation
    scores = knn_cv(Xtr, ytr)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[k, scores[k]] for k in sorted(scores)])
    k_best = best_k(scores)
    pred, _ = knn_predict(Xtr, ytr, Xte, k_best)
    tp, fp, tn, fn = confusion(pred, yte)
    m = metrics(tp, fp, tn, fn)
    out += [("knn_best_k", k_best), ("knn_tp", tp), ("knn_fp", fp), ("knn_tn", tn),
            ("knn_fn", fn), ("knn_accuracy", m["accuracy"]),
            ("knn_precision", m["precision"]), ("knn_recall", m["recall"]),
            ("knn_f1", m["f1"])]

    # 2.8 alert threshold chosen with money on the training rows
    best_tau, best_cost = None, None
    for k in range(1, 20):
        tau = k / 20
        _, fp, _, fn = confusion(p_tr >= tau, ytr)
        c = money_cost(fp, fn)
        if best_cost is None or c < best_cost:
            best_tau, best_cost = tau, c
    _, fp, _, fn = confusion(p_te >= best_tau, yte)
    cost_test_tau = money_cost(fp, fn)
    _, fp, _, fn = confusion(p_te >= 0.5, yte)
    cost_test_05 = money_cost(fp, fn)
    out += [("cost_tau", float(best_tau)), ("cost_train_at_tau", int(best_cost)),
            ("cost_test_at_tau", int(cost_test_tau)), ("cost_test_at_0.5", int(cost_test_05)),
            ("cost_test_flag_nobody", int(money_cost(0, int(yte.sum()))))]

    write_results("part2_results.txt", out)


if __name__ == "__main__":
    main()
