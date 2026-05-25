import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)
        self.lb = None
        self.ub = None
        self.mean = None
        self.sigma = None
        self.C = None
        self.best_x = None
        self.best_val = np.inf
        self.evals = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.mean = self.rng.uniform(self.lb, self.ub)
        self.sigma = 0.2 * (self.ub - self.lb).mean()
        self.C = np.eye(self.dim)
        val = func(self.mean)
        self.evals += 1
        if val < self.best_val:
            self.best_val = val
            self.best_x = self.mean.copy()
            report_best(self.best_val, self.best_x)

        lam = 4 + int(3 * np.log(self.dim))
        lam = min(lam, self.budget - self.evals)
        if lam < 2:
            lam = 2
        mu = lam // 2
        if mu == 0:
            mu = 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        cc = 4.0 / (self.dim + 4.0)
        cs = (mu + 2.0) / (self.dim + mu + 5.0)
        c1 = 2.0 / ((self.dim + 1.3) ** 2 + mu)
        cmu = min(1 - c1, 2.0 * (mu - 2.0 + 1.0 / mu) / ((self.dim + 2.0) ** 2 + mu))
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (self.dim + 1)) - 1) + cs
        chiN = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        while self.evals < self.budget:
            A = np.linalg.cholesky(self.C)
            samples = self.rng.randn(lam, self.dim)
            X = self.mean + self.sigma * (samples @ A.T)
            X = np.clip(X, self.lb, self.ub)
            vals = np.array([func(x) for x in X])
            self.evals += lam
            for i in range(lam):
                if vals[i] < self.best_val:
                    self.best_val = vals[i]
                    self.best_x = X[i].copy()
                    report_best(self.best_val, self.best_x)
            if self.evals >= self.budget:
                break

            idx = np.argsort(vals)
            X_sorted = X[idx]
            old_mean = self.mean.copy()
            self.mean = (weights * X_sorted[:mu].T).sum(axis=1)

            z = (self.mean - old_mean) / self.sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu) * np.linalg.solve(self.C, z)
            ps_norm = np.linalg.norm(ps)
            self.sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))

            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu) * z
            self.C = (1 - c1 - cmu) * self.C + c1 * np.outer(pc, pc)
            for i in range(mu):
                self.C += cmu * weights[i] * np.outer((X_sorted[i] - old_mean) / self.sigma, (X_sorted[i] - old_mean) / self.sigma)
            self.C = (self.C + self.C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-20)
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # adjust lam for next iteration if budget low
            lam = min(lam, self.budget - self.evals)
            if lam < 2:
                break

        return self.best_val, self.best_x