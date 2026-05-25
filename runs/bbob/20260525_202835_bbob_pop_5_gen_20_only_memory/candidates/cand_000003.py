import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_value = np.inf
        self.evals = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial design: 2*dim random points
        n_initial = min(2 * self.dim, self.budget)
        for _ in range(n_initial):
            if self.evals >= self.budget:
                break
            x = self.rng.uniform(lb, ub)
            val = func(x)
            self.evals += 1
            if val < self.best_value:
                self.best_value = val
                self.best_x = x.copy()
                report_best(val, x)
        # Adaptive search with restart
        radius = 0.2 * (ub - lb)  # initial radius per dimension
        stagnation = 0
        max_stagnation = 10 * self.dim
        while self.evals < self.budget:
            # Sample candidate around best
            candidate = self.best_x + radius * self.rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            self.evals += 1
            if val < self.best_value:
                self.best_value = val
                self.best_x = candidate.copy()
                report_best(val, candidate)
                stagnation = 0
                radius = np.clip(radius * 1.2, 1e-5 * (ub - lb), 0.5 * (ub - lb))
            else:
                stagnation += 1
                radius = np.clip(radius * 0.9, 1e-5 * (ub - lb), 0.5 * (ub - lb))
            if stagnation >= max_stagnation:
                # Restart: sample new point uniformly
                x_new = self.rng.uniform(lb, ub)
                val_new = func(x_new)
                self.evals += 1
                if val_new < self.best_value:
                    self.best_value = val_new
                    self.best_x = x_new.copy()
                    report_best(val_new, x_new)
                radius = 0.2 * (ub - lb)
                stagnation = 0
        return self.best_value, self.best_x