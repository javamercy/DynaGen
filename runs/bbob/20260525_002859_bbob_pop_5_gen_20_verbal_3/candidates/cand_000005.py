import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_val = np.inf
        best_x = None
        evals = 0

        # Global exploration: random uniform sampling
        n_global = max(1, int(0.3 * self.budget))
        for _ in range(n_global):
            if evals >= self.budget:
                break
            x = self.rng.uniform(lb, ub, size=self.dim)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Local refinement: Gaussian perturbations around best point
        step = 0.1 * np.mean(ub - lb)
        while evals < self.budget:
            x = best_x + self.rng.normal(0, step, size=self.dim)
            x = np.clip(x, lb, ub)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
                step *= 0.9  # reduce step on improvement
        return best_val, best_x