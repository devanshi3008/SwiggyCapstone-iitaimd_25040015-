"""Part 3 - Segmentation and dimensionality reduction (all 800 rows, X1..X5)."""
import numpy as np

from common import load_data, col_mean_sd, write_results, write_csv


# ---------------------------------------------------------------- k-means
def kmeans(Z, k):
    """Pinned K-means: start rows 0, n//k, 2*(n//k), ...; stop at first pass with no change."""
    n = Z.shape[0]
    step = n // k
    cent = Z[[i * step for i in range(k)]].copy()
    labels = None
    passes = 0
    while True:
        d2 = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)            # argmin returns the lowest index on ties
        passes += 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = Z[labels == j]
            if members.shape[0] > 0:
                cent[j] = members.sum(axis=0) / members.shape[0]
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, cent, wcss, passes


# ---------------------------------------------------------------- hierarchical
NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
DIST = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
])


def linkage_distance(a, b, rule):
    pair = [DIST[i, j] for i in a for j in b]
    if rule == "single":
        return min(pair)
    if rule == "complete":
        return max(pair)
    return sum(pair) / len(pair)


def label(cluster):
    return "+".join(NAMES[i] for i in sorted(cluster))


def agglomerate(rule):
    clusters = [[i] for i in range(len(NAMES))]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_distance(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = sorted([clusters[i], clusters[j]], key=min)
        merges.append((label(a), label(b), d, sorted(a + b)))
        merged = sorted(a + b)
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)] + [merged]
    return merges


def clusters_at_cut(merges):
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
    g = int(np.argmax(gaps))              # cut lies between merge g+1 and merge g+2
    merges_below = g + 1
    return len(NAMES) - merges_below, merges_below


def members_at_cut(merges, merges_below):
    clusters = [[i] for i in range(len(NAMES))]
    for m in merges[:merges_below]:
        members = m[3]
        clusters = [c for c in clusters if not set(c) <= set(members)] + [members]
    return sorted((label(c) for c in clusters), key=lambda s: int(s.split("+")[0][1:]))


# ---------------------------------------------------------------- main
def main():
    X, y, v = load_data()
    C = X[:, :5]
    mu, sd = col_mean_sd(C)
    Z = (C - mu) / sd
    out = []

    # 3.1 K-means for K = 2..6
    wcss, fits = {}, {}
    for k in range(2, 7):
        labels, cent, w, passes = kmeans(Z, k)
        wcss[k] = w
        fits[k] = labels
        out.append((f"wcss_k{k}", w))

    # 3.2 elbow rule
    elbow = None
    for k in (3, 4, 5):
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
            elbow = k
            break
    out.append(("elbow_k", elbow if elbow is not None else "none"))

    # 3.3 profile K = 3 segments ordered by mean X3
    labels = fits[3]
    order = sorted(range(3), key=lambda j: C[labels == j, 2].mean())
    seg_of = {}
    rows = []
    for name, j in zip("ABC", order):
        mask = labels == j
        seg_of[name] = mask
        out.append((f"size_segment_{name}", int(mask.sum())))
        means = C[mask].sum(axis=0) / mask.sum()
        rows.append([name, int(mask.sum())] + list(means)
                    + [y[mask].sum() / mask.sum(), v[mask].sum() / mask.sum()])
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
               "outcome_rate", "mean_value"], rows)

    # 3.4 hierarchical clustering on the supplied table
    hrows = []
    cut_info = {}
    for rule in ("single", "complete", "average"):
        merges = agglomerate(rule)
        for no, (a, b, h, _) in enumerate(merges, start=1):
            hrows.append([rule, no, a, b, h])
        n_clusters, below = clusters_at_cut(merges)
        cut_info[rule] = (n_clusters, members_at_cut(merges, below))
    write_csv("part3_hierarchical.csv",
              ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)
    for rule in ("single", "complete", "average"):
        out.append((f"{rule}_clusters_at_cut", cut_info[rule][0]))

    # 3.5 PCA in seven steps
    n = Z.shape[0]
    mean_vec = Z.sum(axis=0) / n                     # step 1 (zeros up to rounding)
    Xc = Z - mean_vec                                # step 2
    S = (Xc.T @ Xc) / (n - 1)                        # step 3
    evals, evecs = np.linalg.eigh(S)                 # step 4
    idx = np.argsort(evals)[::-1]                    # step 5
    evals, evecs = evals[idx], evecs[:, idx]
    for j in range(evecs.shape[1]):                  # sign convention
        if evecs[np.argmax(np.abs(evecs[:, j])), j] < 0:
            evecs[:, j] = -evecs[:, j]
    W = evecs[:, :2]                                 # step 6
    P = Xc @ W                                       # step 7
    total = float(evals.sum())
    ratios = evals / total
    cum = np.cumsum(ratios)
    out.append(("eigenvalue_sum", total))
    for j in range(5):
        out.append((f"eigenvalue_{j + 1}", float(evals[j])))
    for j in range(5):
        out.append((f"explained_ratio_{j + 1}", float(ratios[j])))
    for j in range(5):
        out.append((f"cumulative_ratio_{j + 1}", float(cum[j])))
    out.append(("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1))
    for pc in (0, 1):
        for j in range(5):
            out.append((f"pc{pc + 1}_loading_X{j + 1}", float(W[j, pc])))
    for r in range(3):
        out.append((f"row{r + 1}_pc1", float(P[r, 0])))
        out.append((f"row{r + 1}_pc2", float(P[r, 1])))
    for name in "ABC":
        m = seg_of[name]
        out.append((f"mean_pc1_segment_{name}", float(P[m, 0].sum() / m.sum())))

    write_results("part3_results.txt", out)
    # cluster membership at the cut is printed for the README (not a graded file)
    for rule in ("single", "complete", "average"):
        print(rule, cut_info[rule][0], " | ".join(cut_info[rule][1]))


if __name__ == "__main__":
    main()
