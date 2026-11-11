"""Part 4 - Model choice: splits, voting, AdaBoost by hand and network sizing."""
import math

import numpy as np

from common import (prepare_split, fit_part2_logistic, sigmoid, confusion, metrics,
                    knn_cv, knn_predict, best_k, write_results, write_csv)


# ---------------------------------------------------------------- 4.1 impurity
def entropy(pos, n):
    if n == 0:
        return 0.0
    h = 0.0
    for c in (pos, n - pos):
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def gini(pos, n):
    if n == 0:
        return 0.0
    p1 = pos / n
    return 1.0 - p1 * p1 - (1 - p1) * (1 - p1)


def split_conditions(Xraw):
    return {
        "S1": Xraw[:, 5] == 1,        # X6 = 1
        "S2": Xraw[:, 6] == 1,        # X7 = 1
        "S3": Xraw[:, 2] > 0.5,       # X3 > 0.5
        "S4": Xraw[:, 0] <= 24,       # X1 <= 24
    }


# ---------------------------------------------------------------- 4.5 sizing helpers
def dense_params(n_in, n_out):
    return n_in * n_out + n_out


def conv_params(k, c_in, filters):
    return k * k * c_in * filters + filters


def out_side(side, f, pad, stride):
    return (side - f + 2 * pad) // stride + 1


def main():
    S = prepare_split()
    Xtr_raw, ytr = S["Xtr_raw"], S["ytr"]
    Xte_raw, yte = S["Xte_raw"], S["yte"]
    out = []

    # 4.1 which split is purest?
    n, P = ytr.size, int(ytr.sum())
    root_h, root_g = entropy(P, n), gini(P, n)
    out += [("root_entropy", root_h), ("root_gini", root_g)]
    conds = split_conditions(Xtr_raw)
    rows, ig, gg = [], {}, {}
    for name, mask in conds.items():
        ny, py = int(mask.sum()), int(ytr[mask].sum())
        nn, pn = n - ny, P - py
        ig[name] = root_h - (ny / n) * entropy(py, ny) - (nn / n) * entropy(pn, nn)
        gg[name] = root_g - (ny / n) * gini(py, ny) - (nn / n) * gini(pn, nn)
        rows.append([name, ny, py, nn, pn, ig[name], gg[name]])
    write_csv("part4_splits.csv",
              ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
              rows)
    best_ig = max(ig, key=lambda k: (ig[k], -int(k[1])))
    best_gg = max(gg, key=lambda k: (gg[k], -int(k[1])))
    out += [("best_split_by_info_gain", best_ig), ("best_split_by_gini_gain", best_gg)]

    # 4.2 stump, logistic, KNN and two votes on the test rows
    m_tr = conds[best_ig]
    rate_yes = ytr[m_tr].sum() / m_tr.sum()
    rate_no = ytr[~m_tr].sum() / (~m_tr).sum()
    out += [("stump_rate_yes", float(rate_yes)), ("stump_rate_no", float(rate_no))]
    m_te = split_conditions(Xte_raw)[best_ig]
    p_stump = np.where(m_te, rate_yes, rate_no)

    w, b, _ = fit_part2_logistic(S)
    p_log = sigmoid(S["Xte"] @ w + b)
    k = best_k(knn_cv(S["Xtr"], ytr))
    pred_knn, p_knn = knn_predict(S["Xtr"], ytr, S["Xte"], k)

    preds = {
        "logistic": (p_log >= 0.3).astype(int),
        "knn": pred_knn,
        "stump": (p_stump >= 0.3).astype(int),
    }
    votes = preds["logistic"] + preds["knn"] + preds["stump"]
    preds["hard_vote"] = (votes >= 2).astype(int)
    preds["soft_vote"] = ((p_log + p_knn + p_stump) / 3 >= 0.3).astype(int)
    vrows = []
    for name in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
        tp, fp, tn, fn = confusion(preds[name], yte)
        m = metrics(tp, fp, tn, fn)
        vrows.append([name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]])
    write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
              vrows)

    # 4.3 two rounds of AdaBoost by hand
    yb = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h = [np.array([1, 1, -1, -1, -1, -1, -1, -1]),
         np.array([-1, 1, 1, -1, 1, -1, 1, -1])]
    wts = np.full(8, 1 / 8)
    alphas, arows = [], []
    for t in range(2):
        eps = float(wts[h[t] != yb].sum())
        alpha = 0.5 * math.log((1 - eps) / eps)
        unnorm = wts * np.exp(-alpha * yb * h[t])
        new = unnorm / unnorm.sum()
        for i in range(8):
            arows.append([t + 1, i + 1, wts[i], unnorm[i], new[i]])
        out += [(f"round{t + 1}_error", eps), (f"round{t + 1}_alpha", alpha)]
        alphas.append(alpha)
        wts = new
    ens = np.sign(alphas[0] * h[0] + alphas[1] * h[1])
    out.append(("ensemble_accuracy", float((ens == yb).sum() / 8)))
    write_csv("part4_adaboost.csv",
              ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], arows)

    # 4.4 out-of-bag share
    out.append(("oob_fraction", (1 - 1 / 560) ** 560))

    # 4.5 network sizing (counting only; nothing is trained)
    mlp = [("mlp_hidden1", dense_params(784, 16)),
           ("mlp_hidden2", dense_params(16, 16)),
           ("mlp_output", dense_params(16, 10))]
    side = out_side(28, 5, 2, 1)            # conv1 -> 28
    side = out_side(side, 2, 0, 2)          # pool  -> 14
    side = out_side(side, 5, 0, 1)          # conv2 -> 10
    side = out_side(side, 2, 0, 2)          # pool  -> 5
    flat = side * side * 16
    cnn = [("cnn_conv1", conv_params(5, 1, 6)),
           ("cnn_conv2", conv_params(5, 6, 16)),
           ("cnn_dense1", dense_params(flat, 32)),
           ("cnn_dense2", dense_params(32, 10))]
    write_csv("part4_architecture.csv", ["layer", "parameters"], [list(r) for r in mlp + cnn])
    out += [("mlp_total", sum(p for _, p in mlp)), ("cnn_total", sum(p for _, p in cnn))]
    out += [("fc_layer_params", dense_params(128 * 128 * 3, 1024)),
            ("conv_layer_params", conv_params(3, 3, 64))]
    d_in, d_h, d_out = 64, 128, 16
    rnn_total = d_h * d_in + d_h * d_h + d_out * d_h + d_h + d_out
    out.append(("rnn_total", rnn_total))

    write_results("part4_results.txt", out)


if __name__ == "__main__":
    main()
