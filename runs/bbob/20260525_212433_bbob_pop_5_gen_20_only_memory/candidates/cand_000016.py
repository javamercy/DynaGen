import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if budget >= 100:
            self.lambda_ = max(4, int(4 + 3 * np.log(dim)))
        else:
            self.lambda_ = max(4, int(budget / (dim + 1)))
        self.mu = self.lambda_ // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights**2)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * self.mueff - 1.0) / ((self.dim + 2.0)**2 + self.mueff)
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        self.sigma = 0.5
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.ones(dim)
        self.chol_diag = np.ones(dim)
        self.mean = None
        self.best_value = np.inf
        self.best_x = np.zeros(dim)
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        N_init = max(4, min(100, self.budget // 10))
        if N_init > self.budget - 2:
            N_init = self.budget // 2
        x_init = lb + (ub - lb) * self.rng.uniform(size=(N_init, self.dim))
        vals = np.array([func(x) for x in x_init])
        self.evaluations = N_init
        best_idx = np.argmin(vals)
        self.best_value = vals[best_idx]
        self.best_x = x_init[best_idx].copy()
        self.mean = self.best_x.copy()
        report_best(self.best_value, self.best_x)
        while self.evaluations + self.lambda_ <= self.budget:
            z = self.rng.normal(0, 1, size=(self.lambda_, self.dim))
            x = self.mean + self.sigma * (z * np.sqrt(self.C))
            x = np.clip(x, lb, ub)
            vals = np.array([func(x[i]) for i in range(self.lambda_)])
            self.evaluations += self.lambda_
            idx = np.argsort(vals)
            x_sorted = x[idx]
            vals_sorted = vals[idx]
            if vals_sorted[0] < self.best_value:
                self.best_value = vals_sorted[0]
                self.best_x = x_sorted[0].copy()
                report_best(self.best_value, self.best_x)
            old_mean = self.mean.copy()
            self.mean = np.dot(self.weights, x_sorted[:self.mu])
            zmean = np.dot(self.weights, z[idx[:self.mu]])
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zmean
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ((self.mean - old_mean) / self.sigma)
            dev = (x_sorted[:self.mu] - old_mean) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc**2) + self.cmu * np.dot(self.weights, dev**2)
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
            self.C = np.maximum(self.C, 1e-20)
        return self.best_value, self.best_x