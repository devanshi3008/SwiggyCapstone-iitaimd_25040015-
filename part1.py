"""Part 1 - Estimation, inference and a regression baseline (all 800 rows)."""
import math

import numpy as np

from common import load_data, mean, sample_sd, sample_var, sample_cov, write_results


def simple_ols(x, y):
    """Closed-form one-predictor regression with divisor n - 1 for cov and var."""
    slope = sample_cov(x, y) / sample_var(x)
    intercept = mean(y) - slope * mean(x)
    fitted = intercept + slope * x
    ybar = mean(y)
    tss = float(((y - ybar) ** 2).sum())
    ess = float(((fitted - ybar) ** 2).sum())
    rss = float(((y - fitted) ** 2).sum())
    n = y.size
    return {
        "slope": slope, "intercept": intercept, "fitted": fitted,
        "tss": tss, "ess": ess, "rss": rss,
        "r2": 1.0 - rss / tss,
        "rmse": math.sqrt(rss / n),
        "mae": float(np.abs(y - fitted).sum() / n),
    }


def split_resid_sd(fitted, resid):
    """Sample SD of residuals in the lower and upper halves by fitted value (stable sort)."""
    order = np.argsort(fitted, kind="stable")
    half = order.size // 2
    lower = sample_sd(resid[order[:half]])
    upper = sample_sd(resid[order[half:]])
    return lower, upper, upper / lower


def adj_r2(r2, n, p):
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def main():
    X, _, v = load_data()
    n = v.size
    x4 = X[:, 3]
    out = []

    # 1.1 estimate with an interval
    mean_value = mean(v)
    sd_value = sample_sd(v)
    se_value = sd_value / math.sqrt(n)
    out += [("n", n), ("mean_value", mean_value), ("sd_value", sd_value),
            ("se_value", se_value),
            ("ci95_low", mean_value - 1.96 * se_value),
            ("ci95_high", mean_value + 1.96 * se_value)]

    # 1.2 two-tailed z test of mu = 8.0
    z_stat = (mean_value - 8.0) / se_value
    out += [("z_stat", z_stat), ("reject_h0", abs(z_stat) > 1.96)]

    # 1.3 sample size for margin E = 0.5
    target = (1.96 * sd_value / 0.5) ** 2
    n_required = math.ceil(target)
    out += [("n_required", n_required)]

    # 1.4 simple regression value ~ X4
    m = simple_ols(x4, v)
    out += [("reg_slope", m["slope"]), ("reg_intercept", m["intercept"]),
            ("tss", m["tss"]), ("ess", m["ess"]), ("rss", m["rss"]),
            ("tss_equals_ess_plus_rss", abs(m["tss"] - m["ess"] - m["rss"]) < 1e-6 * m["tss"]),
            ("r2", m["r2"]), ("rmse", m["rmse"]), ("mae", m["mae"])]

    # 1.5 gradient descent on standardised X4
    mu4, sd4 = mean(x4), sample_sd(x4)
    z = (x4 - mu4) / sd4
    slope_m, c = 0.0, 0.0
    eta = 0.1

    def cost(mm, cc):
        e = v - (mm * z + cc)
        return float((e * e).sum() / n)

    gd = {0: cost(slope_m, c)}
    for t in range(1, 501):
        e = v - (slope_m * z + c)
        grad_m = -2.0 / n * float((z * e).sum())
        grad_c = -2.0 / n * float(e.sum())
        slope_m -= eta * grad_m
        c -= eta * grad_c
        if t in (1, 10, 100, 500):
            gd[t] = cost(slope_m, c)
    gd_slope = slope_m / sd4
    gd_intercept = c - slope_m * mu4 / sd4
    out += [("gd_J0", gd[0]), ("gd_J1", gd[1]), ("gd_J10", gd[10]),
            ("gd_J100", gd[100]), ("gd_J500", gd[500]),
            ("gd_slope", gd_slope), ("gd_intercept", gd_intercept),
            ("gd_slope_gap", abs(gd_slope - m["slope"]))]

    # 1.6 funnel check, then the log-log remedy
    lo, up, ratio = split_resid_sd(m["fitted"], v - m["fitted"])
    out += [("resid_sd_lower_raw", lo), ("resid_sd_upper_raw", up), ("resid_sd_ratio_raw", ratio)]
    lx, lv = np.log(x4), np.log(v)
    ll = simple_ols(lx, lv)
    out += [("loglog_intercept", ll["intercept"]), ("loglog_slope", ll["slope"]),
            ("loglog_r2", ll["r2"]), ("loglog_rmse", ll["rmse"])]
    lo, up, ratio = split_resid_sd(ll["fitted"], lv - ll["fitted"])
    out += [("resid_sd_lower_log", lo), ("resid_sd_upper_log", up), ("resid_sd_ratio_log", ratio)]

    # 1.7 multiple regression by the normal equation
    D = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    beta = np.linalg.solve(D.T @ D, D.T @ lv)
    fitted = D @ beta
    rss = float(((lv - fitted) ** 2).sum())
    tss = float(((lv - mean(lv)) ** 2).sum())
    mr_r2 = 1.0 - rss / tss
    for j, b in enumerate(beta):
        out.append((f"mr_beta_{j}", float(b)))
    out += [("mr_r2", mr_r2), ("mr_adj_r2", adj_r2(mr_r2, n, 7)),
            ("loglog_adj_r2", adj_r2(ll["r2"], n, 1))]

    write_results("part1_results.txt", out)


if __name__ == "__main__":
    main()
