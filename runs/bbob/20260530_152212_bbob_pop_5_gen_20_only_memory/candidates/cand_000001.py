import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        scale = ub - lb
        # initial mean
        mean = lb + self.rng.rand(dim) * scale
        # initial step size
        sigma = 0.2 * np.mean(scale)
        # recombination weights
        lam = 4 + int(np.floor(3 * np.log(dim)))
        lam = min(lam, self.budget)
        if lam < 2:
            lam = 2
        # weights for recombination
        weights = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        # learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff-1)/(dim+1)) - 1) + cs
        # evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # covariance matrix
        C = np.eye(dim)
        # best tracking
        best_x = mean.copy()
        best_val = func(best_x)
        evals = 1
        self._report(best_val, best_x)
        # main loop
        while evals + lam <= self.budget:
            # sample pop
            A = np.linalg.cholesky(C)
            Z = self.rng.randn(lam, dim)
            X = mean + sigma * (A @ Z.T).T
            # clamp
            X = np.clip(X, lb, ub)
            # evaluate
            vals = np.array([func(x) for x in X])
            evals += lam
            idx = np.argsort(vals)
            if vals[idx[0]] < best_val:
                best_val = vals[idx[0]]
                best_x = X[idx[0]]
                self._report(best_val, best_x)
            # update mean
            old_mean = mean.copy()
            mean = mean + sigma * (A @ (weights[idx] * Z[idx]).sum(axis=0))
            # update evolution paths
            z = A @ (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*evals/lam)) < 1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            # update covariance
            left = (1 - c1 - cmu) * C
            right1 = c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            right2 = cmu * (A @ ((weights[:, None] * Z[idx].T) @ Z[idx]) @ A.T)
            C = left + right1 + right2
            # step size adaptation
            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
        # remaining budget: evaluate additional random points if any
        while evals < self.budget:
            x = lb + self.rng.rand(dim) * scale
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x
                self._report(best_val, best_x)
        return best_val, best_x

    def _report(self, val, x):
        try:
            report_best(val, x)
        except NameError:
            pass