"""Shared helpers used by part1.py to part4.py.

Importing this module only defines functions; it never runs a Part.
"""
import csv
import os

import numpy as np

DATA_PATH = os.path.join("data", "records.csv")
OUT_DIR = "outputs"
FEATURES = [
    "customer_tenure_months",       # X1
    "earlier_ticket_count_90d",     # X2
    "sla_consumed_ratio",           # X3
    "account_value_inr_thousands",  # X4
    "satisfaction_score",           # X5
    "reopened_flag",                # X6
    "self_service_first_flag",      # X7
]
OUTCOME = "escalated"
VALUE = "resolution_minutes"


# ---------------------------------------------------------------- data loading
def load_data():
    """Return (X, y, v): X is 800 x 7 (X1..X7), y the outcome (int), v the value."""
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([int(r[OUTCOME]) for r in rows])
    v = np.array([float(r[VALUE]) for r in rows])
    return X, y, v


# ---------------------------------------------------------------- basic statistics
def mean(a):
    a = np.asarray(a, dtype=float)
    return a.sum() / a.size


def sample_var(a):
    a = np.asarray(a, dtype=float)
    d = a - mean(a)
    return (d * d).sum() / (a.size - 1)


def sample_sd(a):
    return float(np.sqrt(sample_var(a)))


def sample_cov(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return ((a - mean(a)) * (b - mean(b))).sum() / (a.size - 1)


def col_mean_sd(M):
    """Column means and sample SDs of a 2-D array."""
    M = np.asarray(M, dtype=float)
    mu = M.sum(axis=0) / M.shape[0]
    sd = np.sqrt(((M - mu) ** 2).sum(axis=0) / (M.shape[0] - 1))
    return mu, sd


# ---------------------------------------------------------------- Part 2 split
def stratified_split(y):
    """Task 2.1: first floor(7*count/10) rows of each class (file order) -> train."""
    train_mask = np.zeros(y.size, dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]          # already in file order
        n_train = (7 * idx.size) // 10
        train_mask[idx[:n_train]] = True
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    return train_idx, test_idx


def prepare_split():
    """Load data, split, and standardise X1..X7 with training mean / sample SD."""
    X, y, v = load_data()
    tr, te = stratified_split(y)
    mu, sd = col_mean_sd(X[tr])
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    return {
        "X": X, "y": y, "v": v, "train_idx": tr, "test_idx": te,
        "Xtr": Xtr, "ytr": y[tr], "Xte": Xte, "yte": y[te],
        "Xtr_raw": X[tr], "Xte_raw": X[te],
    }


# ---------------------------------------------------------------- logistic regression
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(p, y):
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() / y.size)


def train_logistic(X, y, eta, n_updates, record=()):
    """Full-batch gradient descent from zero. Returns (w, b, {epoch: loss})."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(sigmoid(X @ w + b), y)
    for t in range(1, n_updates + 1):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n
        grad_b = err.sum() / n
        w = w - eta * grad_w
        b = b - eta * grad_b
        if t in record:
            losses[t] = bce(sigmoid(X @ w + b), y)
    return w, b, losses


def fit_part2_logistic(S):
    """The Task 2.2 model (eta 0.1, 1000 updates) on the standardised training rows."""
    return train_logistic(S["Xtr"], S["ytr"], 0.1, 1000, record=(0, 1, 10, 100, 1000))


# ---------------------------------------------------------------- metrics
def confusion(pred, y):
    pred = np.asarray(pred).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "specificity": spec, "fpr": fpr, "f1": f1}


# ---------------------------------------------------------------- k-nearest neighbours
def knn_share(Xref, yref, Xq, k):
    """Share of positive neighbours among the K nearest reference rows for each query.
    Neighbours ranked by (distance, reference position) using a stable sort."""
    shares = np.empty(Xq.shape[0])
    for i in range(Xq.shape[0]):
        diff = Xref - Xq[i]
        dist = np.sqrt((diff * diff).sum(axis=1))
        order = np.argsort(dist, kind="stable")[:k]
        shares[i] = yref[order].sum() / k
    return shares


def knn_predict(Xref, yref, Xq, k):
    """Positive when at least (K + 1) / 2 of the K neighbours are positive."""
    shares = knn_share(Xref, yref, Xq, k)
    votes = np.rint(shares * k)
    return (votes >= (k + 1) / 2).astype(int), shares


def knn_cv(Xtr, ytr, ks=range(1, 22, 2), n_folds=5):
    """5-fold CV by position modulo 5; returns {K: mean F1 over folds}."""
    folds = np.arange(Xtr.shape[0]) % n_folds
    scores = {}
    for k in ks:
        f1s = []
        for f in range(n_folds):
            ref = folds != f
            hold = folds == f
            pred, _ = knn_predict(Xtr[ref], ytr[ref], Xtr[hold], k)
            f1s.append(metrics(*confusion(pred, ytr[hold]))["f1"])
        scores[k] = sum(f1s) / n_folds
    return scores


def best_k(scores):
    best = None
    for k in sorted(scores):
        if best is None or scores[k] > scores[best]:
            best = k
    return best


# ---------------------------------------------------------------- output helpers
def fmt(value):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_results(path, items):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, path), "w", newline="") as f:
        for key, value in items:
            f.write(f"{key} = {fmt(value)}\n")


def write_csv(path, header, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, path), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(x) for x in r])
