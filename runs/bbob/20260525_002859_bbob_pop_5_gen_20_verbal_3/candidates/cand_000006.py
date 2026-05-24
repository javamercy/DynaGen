import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Global exploration phase (10% budget)
        n_global = max(1, int(0.1 * budget))
        for _ in range(n_global):
            if evals >= budget:
                break
            x = rng.uniform(lb, ub)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Adaptive random search with restarts
        radius = (ub - lb).mean() / 5.0
        stagnation = 0
        max_stagnation = max(10, 2 * dim)

        while evals < budget:
            if stagnation >= max_stagnation:
                x = rng.uniform(lb, ub)
                radius = (ub - lb).mean() / 5.0
                stagnation = 0
            else:
                x = best_x + radius * rng.randn(dim)
                x = np.clip(x, lb, ub)

            if evals >= budget:
                break
            val = func(x)
            evals += 1

            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
                radius *= 1.2
                stagnation = 0
            else:
                radius *= 0.8
                stagnation += 1

            radius = max(radius, 1e-10 * (ub - lb).mean())

        return best_val, best_x