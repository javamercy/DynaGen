import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # population size: heuristic
        if budget >= 100:
            self.lambda_ = max(4, int(4 + 3 * np.log(dim)))
        else:
            self.lambda_ = max(4, int(budget / (dim + 1)))
        self.mu = self.lambda_ // 2
        # weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights**2)
        # adaptation parameters
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * self.mueff - 1.0) / ((self.dim + 2.0)**2 + self.mueff)
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        # initialize state
        self.mean = None  # will be set after first evaluation
        self.sigma = 0.5  # step size
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.ones(dim)  # diagonal cov (variance vector)
        self.chol_diag = np.ones(dim)
        self.best_value = np.inf
        self.best_x = np.zeros(dim)
        self.evaluations = 0

    def __call__(self, func):
        # initial mean from bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.mean = lb + (ub - lb) * self.rng.uniform(size=self.dim)
        # evaluate initial mean
        val = func(self.mean)
        self.evaluations += 1
        if val < self.best_value:
            self.best_value = val
            self.best_x = self.mean.copy()
            report_best(val, self.best_x)

        # main loop
        while self.evaluations + self.lambda_ <= self.budget:
            # sample candidates
            z = self.rng.normal(0, 1, size=(self.lambda_, self.dim))
            x = self.mean + self.sigma * (z * np.sqrt(self.C))
            # clip
            x = np.clip(x, lb, ub)
            # evaluate
            vals = np.array([func(x[i]) for i in range(self.lambda_)])
            self.evaluations += self.lambda_
            # sort
            idx = np.argsort(vals)
            x_sorted = x[idx]
            vals_sorted = vals[idx]
            # update best
            if vals_sorted[0] < self.best_value:
                self.best_value = vals_sorted[0]
                self.best_x = x_sorted[0].copy()
                report_best(self.best_value, self.best_x)
            # update mean
            old_mean = self.mean.copy()
            self.mean = np.dot(self.weights, x_sorted[:self.mu])
            # update evolution paths
            zmean = np.dot(self.weights, z[idx[:self.mu]])  # weighted mean of z
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zmean
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ((self.mean - old_mean) / self.sigma)
            # update diagonal covariance
            # compute weighted deviations in x space
            dev = (x_sorted[:self.mu] - old_mean) / self.sigma
            # rank-one update from pc
            C_update = self.c1 * (self.pc**2) + self.cmu * np.dot(self.weights, dev**2)
            self.C = (1 - self.c1 - self.cmu) * self.C + C_update
            # step size control
            sigma_update = np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
            self.sigma *= sigma_update
            # ensure stability
            self.C = np.maximum(self.C, 1e-20)
        # final return
        return self.best_value, self.best_x