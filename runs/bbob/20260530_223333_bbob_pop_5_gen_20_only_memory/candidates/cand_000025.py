import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.lam = int(4 + 3 * np.log(dim))
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1)/(dim + 1)) - 1) + self.cs
        self.mean = None
        self.sigma = 0.5
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.best_val = np.inf
        self.best_x = None
        self.stagnation_count = 0
        self.max_stagnation = 10 + int(5 * np.log(dim))
        self.restart_count = 0

    def __call__(self, func):
        self.lb = func.bounds.lb.copy()
        self.ub = func.bounds.ub.copy()
        np.random.seed(self.seed)
        self._init_restart()
        self.best_val = np.inf
        self.best_x = np.empty(self.dim)
        evals = 0
        while evals < self.budget:
            if self.mean is None:
                self._init_restart()
            # evaluate current mean if not yet
            if self.best_x is None:
                val = func(self.mean)
                evals += 1
                self.best_val = val
                self.best_x = self.mean.copy()
                report_best(self.best_val, self.best_x)
            # check budget for a generation
            if evals + self.lam > self.budget:
                break
            # sample
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.diag(np.sqrt(eigvals))
                B = eigvecs
            except np.linalg.LinAlgError:
                # fallback to identity
                B = np.eye(self.dim)
                D = np.eye(self.dim)
            Z = np.random.randn(self.dim, self.lam)
            X = self.mean[:, np.newaxis] + self.sigma * (B @ D @ Z)
            X = np.clip(X, self.lb[:, np.newaxis], self.ub[:, np.newaxis])
            F = np.empty(self.lam)
            for i in range(self.lam):
                F[i] = func(X[:, i])
                evals += 1
            idx = np.argsort(F)
            F = F[idx]
            X = X[:, idx]
            if F[0] < self.best_val:
                self.best_val = F[0]
                self.best_x = X[:, 0].copy()
                report_best(self.best_val, self.best_x)
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            old_mean = self.mean.copy()
            self.mean = X[:, :self.mu] @ self.weights
            zmean = np.linalg.solve(B @ D, self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * zmean
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2*(evals/self.lam))) < (1.4 + 2/(self.dim+1))) * 1.0
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma
            artmp = (X[:, :self.mu] - old_mean[:, np.newaxis]) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (artmp @ np.diag(self.weights) @ artmp.T)
            self.C = (self.C + self.C.T) / 2
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
            if self.stagnation_count >= self.max_stagnation and evals < self.budget - self.lam:
                self._init_restart()
        return self.best_val, self.best_x

    def _init_restart(self):
        self.mean = self.lb + (self.ub - self.lb) * np.random.rand(self.dim)
        self.sigma = 0.5 * (np.random.uniform(0.2, 0.8))
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.stagnation_count = 0
        self.restart_count += 1