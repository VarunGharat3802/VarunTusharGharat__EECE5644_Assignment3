"""
Question 2: GMM model-order selection with 10-fold cross-validation.

True model: 4-component 2-D GMM, components 1 and 2 overlap strongly.
For N in {10, 100, 1000}, repeated R = 100 times:
  - draw N iid samples,
  - for each order M = 1..10, run 10-fold CV: fit GMM by EM on 9 folds,
    score average validation log-likelihood on the held-out fold,
  - select the order with the highest mean validation log-likelihood.
"""
import json
import time
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 11
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# True GMM
# ---------------------------------------------------------------------------
WEIGHTS = np.array([0.20, 0.30, 0.15, 0.35])
MEANS = np.array([[0.0, 0.0], [1.5, 0.5], [5.0, 4.0], [-4.0, 5.0]])
COVS = np.array([
    [[1.0, 0.4], [0.4, 0.8]],
    [[0.8, -0.3], [-0.3, 1.0]],
    [[1.2, 0.5], [0.5, 0.7]],
    [[0.6, 0.0], [0.0, 1.4]],
])


def sample_gmm(n, rng):
    comp = rng.choice(len(WEIGHTS), size=n, p=WEIGHTS)
    X = np.empty((n, 2))
    for k in range(len(WEIGHTS)):
        idx = comp == k
        X[idx] = rng.multivariate_normal(MEANS[k], COVS[k], idx.sum())
    return X, comp


def true_logpdf(X):
    dens = sum(w * mvn(m, c).pdf(X) for w, m, c in zip(WEIGHTS, MEANS, COVS))
    return np.log(dens)


# ---------------------------------------------------------------------------
# K-fold CV for GMM order
# ---------------------------------------------------------------------------
ORDERS = list(range(1, 11))
K_FOLDS = 10


def cv_gmm_order(X, rng):
    n = len(X)
    folds = np.array_split(rng.permutation(n), K_FOLDS)
    scores = np.full(len(ORDERS), -np.inf)
    for i, M in enumerate(ORDERS):
        ll_total, feasible = 0.0, True
        for k in range(K_FOLDS):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(K_FOLDS) if j != k])
            if len(tr) < M:                 # EM cannot fit more components than points
                feasible = False
                break
            gmm = GaussianMixture(n_components=M, covariance_type="full",
                                  n_init=3, max_iter=300, tol=1e-4,
                                  reg_covar=1e-6, init_params="kmeans",
                                  random_state=int(rng.integers(1 << 31)))
            gmm.fit(X[tr])
            ll_total += gmm.score_samples(X[val]).sum()   # validation log-likelihood
        if feasible:
            scores[i] = ll_total / n        # average per-sample validation log-lik
    return ORDERS[int(np.argmax(scores))], scores


if __name__ == "__main__":
    t0 = time.time()
    N_LIST = [10, 100, 1000]
    R = 100

    # Check overlap condition for components 1 and 2
    d12 = float(np.linalg.norm(MEANS[0] - MEANS[1]))
    avg_eig_sum = float(np.mean([np.linalg.eigvalsh(COVS[0]).sum(),
                                 np.linalg.eigvalsh(COVS[1]).sum()]))
    print(f"||mu1-mu2|| = {d12:.3f}, avg eigenvalue sum = {avg_eig_sum:.3f}")

    import os
    sel, curves, true_ll = {}, {}, {}
    for n in N_LIST:
        ck = f"q2_ckpt_N{n}.json"
        if os.path.exists(ck):                       # resume from checkpoint
            d = json.load(open(ck))
        else:
            rng_n = np.random.default_rng([SEED, n])  # independent stream per N
            d = {"sel": [], "curves": [], "true_ll": []}
            for r in range(R):
                X, _ = sample_gmm(n, rng_n)
                best, scores = cv_gmm_order(X, rng_n)
                d["sel"].append(best)
                d["curves"].append([float(v) if np.isfinite(v) else None for v in scores])
                d["true_ll"].append(float(true_logpdf(X).mean()))
            json.dump(d, open(ck, "w"))
        sel[n] = d["sel"]
        curves[n] = [[(-np.inf if v is None else v) for v in c] for c in d["curves"]]
        true_ll[n] = d["true_ll"]
        print(f"N={n:5d} done  [{time.time() - t0:.0f}s]", flush=True)

    counts = {n: [int(np.sum(np.array(sel[n]) == M)) for M in ORDERS] for n in N_LIST}
    out = {"orders": ORDERS, "R": R, "d12": d12, "avg_eig_sum": avg_eig_sum,
           "selections": {str(n): sel[n] for n in N_LIST},
           "counts": {str(n): counts[n] for n in N_LIST},
           "curves": {str(n): [[(None if not np.isfinite(v) else v) for v in c] for c in curves[n]] for n in N_LIST},
           "true_ll": {str(n): true_ll[n] for n in N_LIST}}
    with open("q2_results.json", "w") as f:
        json.dump(out, f)
    for n in N_LIST:
        print(n, counts[n])

    plt.rcParams.update({"font.size": 11})
    # (a) true GMM with samples
    Xs, cs = sample_gmm(1000, rng)
    g1, g2 = np.meshgrid(np.linspace(-7, 9, 300), np.linspace(-3, 9, 300))
    G = np.column_stack([g1.ravel(), g2.ravel()])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.contour(g1, g2, np.exp(true_logpdf(G)).reshape(g1.shape), levels=12, cmap="viridis")
    for k in range(4):
        ax.scatter(*Xs[cs == k].T, s=6, alpha=0.6, label=f"Component {k + 1} ($\\alpha$={WEIGHTS[k]})")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal"); fig.tight_layout(); fig.savefig("figures/q2_true_gmm.pdf")

    # (b) selection rate grouped bars
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.27
    for i, n in enumerate(N_LIST):
        ax.bar(np.array(ORDERS) + (i - 1) * w, np.array(counts[n]) / R * 100, w, label=f"N={n}")
    ax.axvline(4, color="k", ls=":", lw=1)
    ax.set_xticks(ORDERS); ax.set_xlabel("Selected GMM order")
    ax.set_ylabel("Selection rate (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/q2_selection_rates.pdf")

    # (c) mean validation log-likelihood vs order
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, n in zip(axs, N_LIST):
        A = np.array(curves[n])
        A_f = np.where(np.isfinite(A), A, np.nan)
        med = np.nanmedian(A_f, 0)
        q1, q3 = np.nanpercentile(A_f, 25, 0), np.nanpercentile(A_f, 75, 0)
        ax.plot(ORDERS, med, "o-", label="median")
        ax.fill_between(ORDERS, q1, q3, alpha=0.3, label="IQR")
        ax.axhline(np.mean(true_ll[n]), color="r", ls="--", lw=1, label="true pdf")
        ax.set_title(f"N = {n}"); ax.set_xlabel("GMM order"); ax.set_xticks(ORDERS)
        ax.grid(alpha=0.3)
        top = max(np.nanmax(q3), np.mean(true_ll[n])) + 0.4
        ax.set_ylim(top - 6.5, top)
    axs[0].set_ylabel("Avg. validation log-likelihood"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/q2_cv_curves.pdf")

    # (d) boxplot of selected order
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.boxplot([sel[n] for n in N_LIST], tick_labels=[str(n) for n in N_LIST])
    ax.axhline(4, color="r", ls="--", lw=1, label="true order")
    ax.set_xlabel("Dataset size N"); ax.set_ylabel("Selected order")
    ax.set_yticks(ORDERS); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figures/q2_boxplot.pdf")
    print(f"Done in {time.time() - t0:.0f}s")
