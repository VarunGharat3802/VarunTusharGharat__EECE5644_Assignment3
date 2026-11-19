"""
Question 1: MLP approximation of class posteriors + MAP classification.

- C = 4 classes, uniform priors, 3-D Gaussian class-conditionals.
- 2-layer MLP: P hidden Softplus units (smooth ramp), softmax output.
- Training: maximum likelihood (= minimum average cross-entropy) with L-BFGS,
  multiple random restarts, keep the highest training log-likelihood.
- Model order (P) chosen by 10-fold CV minimising validation P(error).
"""
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. True data distribution
# ---------------------------------------------------------------------------
C, D = 4, 3
PRIORS = np.full(C, 1.0 / C)
SCALE = 0.95
MEANS = SCALE * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
COVS = np.array([
    [[1.0, 0.3, 0.1], [0.3, 0.8, 0.0], [0.1, 0.0, 0.6]],
    [[0.7, -0.2, 0.0], [-0.2, 1.1, 0.25], [0.0, 0.25, 0.9]],
    [[1.2, 0.0, -0.3], [0.0, 0.6, 0.1], [-0.3, 0.1, 0.9]],
    [[0.8, 0.25, 0.25], [0.25, 0.8, 0.25], [0.25, 0.25, 0.8]],
])


def generate_data(n, rng):
    labels = rng.choice(C, size=n, p=PRIORS)
    X = np.empty((n, D))
    for c in range(C):
        idx = labels == c
        X[idx] = rng.multivariate_normal(MEANS[c], COVS[c], idx.sum())
    return X, labels


def map_classifier(X):
    """Theoretically optimal (MAP) rule using the true pdf."""
    log_post = np.column_stack(
        [mvn(MEANS[c], COVS[c]).logpdf(X) + np.log(PRIORS[c]) for c in range(C)])
    return log_post.argmax(axis=1)


# ---------------------------------------------------------------------------
# 2. MLP: x -> softplus(W1 x + b1) -> softmax(W2 h + b2)
# ---------------------------------------------------------------------------
def softplus(z):
    return np.logaddexp(0.0, z)          # numerically stable log(1+e^z)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # derivative of softplus, stable


class MLP:
    def __init__(self, P, d=D, c=C):
        self.P, self.d, self.c = P, d, c
        self.shapes = [(P, d), (P,), (c, P), (c,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.theta = None

    def unpack(self, theta):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[i:i + n].reshape(s))
            i += n
        return out

    def init_params(self, rng):
        P, d, c = self.P, self.d, self.c
        W1 = rng.normal(0, np.sqrt(2.0 / (d + P)), (P, d))
        b1 = rng.normal(0, 0.5, P)
        W2 = rng.normal(0, np.sqrt(2.0 / (P + c)), (c, P))
        b2 = np.zeros(c)
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def forward(self, theta, X):
        W1, b1, W2, b2 = self.unpack(theta)
        Z1 = X @ W1.T + b1
        H = softplus(Z1)
        Z2 = H @ W2.T + b2
        log_Y = Z2 - logsumexp(Z2, axis=1, keepdims=True)   # log-softmax
        return Z1, H, log_Y

    def loss_grad(self, theta, X, T):
        """Average cross-entropy (= -avg log-likelihood) and its gradient."""
        N = X.shape[0]
        W1, b1, W2, b2 = self.unpack(theta)
        Z1, H, log_Y = self.forward(theta, X)
        loss = -np.sum(T * log_Y) / N
        dZ2 = (np.exp(log_Y) - T) / N
        dW2 = dZ2.T @ H
        db2 = dZ2.sum(0)
        dZ1 = (dZ2 @ W2) * sigmoid(Z1)
        dW1 = dZ1.T @ X
        db1 = dZ1.sum(0)
        return loss, np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

    def fit(self, X, y, rng, n_restarts=5, maxiter=2000):
        """ML training with random restarts; keep best training log-likelihood."""
        T = np.eye(self.c)[y]
        best = (np.inf, None)
        for _ in range(n_restarts):
            res = minimize(self.loss_grad, self.init_params(rng), args=(X, T),
                           jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "gtol": 1e-7})
            if res.fun < best[0]:
                best = (res.fun, res.x)
        self.train_loss, self.theta = best
        return self

    def predict_proba(self, X):
        return np.exp(self.forward(self.theta, X)[2])

    def predict(self, X):
        return self.forward(self.theta, X)[2].argmax(axis=1)  # MAP decision


def gradient_check(rng):
    """Verify the analytical gradient against central finite differences."""
    m = MLP(5)
    X = rng.normal(size=(20, D))
    T = np.eye(C)[rng.integers(0, C, 20)]
    th = m.init_params(rng)
    _, g = m.loss_grad(th, X, T)
    eps, num = 1e-6, np.zeros_like(th)
    for i in range(th.size):
        e = np.zeros_like(th); e[i] = eps
        num[i] = (m.loss_grad(th + e, X, T)[0] - m.loss_grad(th - e, X, T)[0]) / (2 * eps)
    return np.max(np.abs(num - g)) / max(1e-12, np.max(np.abs(num)))


# ---------------------------------------------------------------------------
# 3. 10-fold cross-validation for P
# ---------------------------------------------------------------------------
def kfold_indices(n, k, rng):
    perm = rng.permutation(n)
    return np.array_split(perm, k)


def cv_select_P(X, y, P_grid, rng, K=10, n_restarts=2, maxiter=500):
    folds = kfold_indices(len(y), K, rng)
    cv_err = []
    for P in P_grid:
        errs = []
        for k in range(K):
            val = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            m = MLP(P).fit(X[tr], y[tr], rng, n_restarts=n_restarts, maxiter=maxiter)
            errs.append(np.mean(m.predict(X[val]) != y[val]))
        cv_err.append(float(np.mean(errs)))
    best_P = P_grid[int(np.argmin(cv_err))]
    return best_P, cv_err


# ---------------------------------------------------------------------------
# 4. Main experiment
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    rel_err = gradient_check(rng)
    print(f"Gradient check relative error: {rel_err:.2e}")

    N_TRAIN = [100, 500, 1000, 5000, 10000]
    P_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

    X_test, y_test = generate_data(100_000, rng)
    train_sets = {n: generate_data(n, rng) for n in N_TRAIN}

    y_opt = map_classifier(X_test)
    p_err_opt = float(np.mean(y_opt != y_test))
    print(f"Theoretically optimal classifier P(error) = {p_err_opt:.4f}")

    results = {"p_err_opt": p_err_opt, "grad_check": rel_err, "P_grid": P_GRID,
               "runs": {}}
    for n in N_TRAIN:
        X, y = train_sets[n]
        best_P, cv_err = cv_select_P(X, y, P_GRID, rng)
        model = MLP(best_P).fit(X, y, rng, n_restarts=10)
        p_err = float(np.mean(model.predict(X_test) != y_test))
        results["runs"][n] = {"best_P": best_P, "cv_err": cv_err,
                              "train_nll": float(model.train_loss), "test_p_err": p_err}
        print(f"N={n:6d}  best P={best_P:3d}  test P(error)={p_err:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        with open("q1_results.json", "w") as f:   # checkpoint after each N
            json.dump(results, f, indent=2)

    with open("q1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Figures --------------------------------------------------------
    plt.rcParams.update({"font.size": 11})
    # (a) test P(error) vs N
    errs = [results["runs"][n]["test_p_err"] for n in N_TRAIN]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogx(N_TRAIN, errs, "o-", lw=2, label="Trained MLP (MAP on MLP posteriors)")
    for n, e in zip(N_TRAIN, errs):
        ax.annotate(f"P={results['runs'][n]['best_P']}", (n, e),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(p_err_opt, color="r", ls="--", lw=1.5,
               label=f"Theoretically optimal ({p_err_opt:.3f})")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Empirical test P(error)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig("figures/q1_perror_vs_n.pdf")

    # (b) CV curves
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n in N_TRAIN:
        ax.plot(P_GRID, results["runs"][n]["cv_err"], "o-", label=f"N={n}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P_GRID); ax.set_xticklabels(P_GRID)
    ax.set_xlabel("Number of hidden perceptrons P")
    ax.set_ylabel("10-fold CV P(error)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/q1_cv_curves.pdf")

    # (c) data scatter
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    Xs, ys = train_sets[1000]
    for c, mk in zip(range(C), "o^sd"):
        ax.scatter(*Xs[ys == c].T, s=8, marker=mk, label=f"Class {c + 1}")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/q1_data.pdf")
    print(f"Done in {time.time() - t0:.0f}s")
