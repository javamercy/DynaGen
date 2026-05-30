import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)
        # CMA-ES parameters
        self.λ = 4 + int(3 * np.log(dim))  # population size
        self.μ = self.λ // 2
        self.weights = np.log(self.μ + 0.5) - np.log(np.arange(1, self.μ + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)
        self.cμ = 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0)**2 + self.mueff)
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        # state
        self.mean = None
        self.sigma = 0.5 * (self.dim)  # initial step size
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.best_value = np.inf
        self.best_x = None
        self.evaluations = 0

    def __call__(self, func):
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        # Initialize mean at random in bounds
        self.mean = lb + self.rng.rand(self.dim) * (ub - lb)
        # Initial incumbent
        val = func(self.mean)
        self.evaluations += 1
        self.best_value = val
        self.best_x = self.mean.copy()
        report_best(self.best_value, self.best_x)
        # Main loop
        while self.evaluations < self.budget:
            # Sample population
            if self.evaluations + self.λ > self.budget:
                # Adjust λ to not exceed budget
                λ_adj = self.budget - self.evaluations
                if λ_adj <= 0:
                    break
            else:
                λ_adj = self.λ
            pop = []
            for _ in range(λ_adj):
                z = self.rng.randn(self.dim)
                x = self.mean + self.sigma * (self.C @ z)
                # Clip to bounds
                x = np.clip(x, lb, ub)
                pop.append(x)
            # Evaluate
            vals = []
            for x in pop:
                if self.evaluations >= self.budget:
                    break
                v = func(x)
                self.evaluations += 1
                vals.append(v)
                if v < self.best_value:
                    self.best_value = v
                    self.best_x = x
                    report_best(self.best_value, self.best_x)
            if len(vals) < 2:
                break
            # Sort
            idx = np.argsort(vals)
            pop_sorted = [pop[i] for i in idx]
            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.zeros(self.dim)
            for i in range(self.μ):
                self.mean += self.weights[i] * pop_sorted[i]
            # Update evolution paths
            zmean = (self.mean - old_mean) / self.sigma
            self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (np.linalg.inv(self.C) @ zmean)
            self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * zmean
            # Update covariance
            artmp = np.zeros((self.dim, self.μ))
            for i in range(self.μ):
                artmp[:, i] = (pop_sorted[i] - old_mean) / self.sigma
            self.C = (1.0 - self.c1 - self.cμ) * self.C + self.c1 * np.outer(self.pc, self.pc)
            for i in range(self.μ):
                self.C += self.cμ * self.weights[i] * np.outer(artmp[:, i], artmp[:, i])
            # Step size update
            self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damps)
        return self.best_value, self.best_x