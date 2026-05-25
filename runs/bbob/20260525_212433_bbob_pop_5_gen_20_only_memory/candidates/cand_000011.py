import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial random point
        x0 = lb + (ub - lb) * self.rng.rand(self.dim)
        val0 = func(x0)
        self.best_val = val0
        self.best_x = x0.copy()
        report_best(self.best_val, self.best_x)
        evals = 1
        sigma = (ub - lb) * 0.2  # step size relative to range
        while evals < self.budget:
            candidate = self.best_x + sigma * self.rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = candidate.copy()
                report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x