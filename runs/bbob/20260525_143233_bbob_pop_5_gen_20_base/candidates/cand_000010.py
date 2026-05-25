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
        width = ub - lb
        mean_width = np.mean(width)

        # initial random point
        best_x = self.rng.uniform(lb, ub, size=self.dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # adaptive parameters
        radius = 0.2 * mean_width
        max_failures = max(5, 2 * self.dim)
        failures = 0

        while evals < self.budget:
            if failures >= max_failures:
                # restart with random point
                x = self.rng.uniform(lb, ub, size=self.dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
                failures = 0
                radius = 0.2 * mean_width
            else:
                # local search around best
                step = self.rng.normal(0, 1, size=self.dim) * (radius * width)
                x = best_x + step
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
                    radius *= 1.1
                    failures = 0
                else:
                    failures += 1
                    radius *= 0.9
                # prevent radius from becoming too small
                if radius < 1e-8 * mean_width:
                    radius = 1e-8 * mean_width
        return best_val, best_x