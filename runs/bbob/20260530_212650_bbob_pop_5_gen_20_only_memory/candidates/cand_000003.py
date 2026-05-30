import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim

        # Initial random point
        x = self.rng.uniform(lb, ub, size=dim)
        val = func(x)
        calls = 1
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)

        # Adaptive radius
        radius = 0.2 * (ub - lb).max()
        stagnation = 0
        stagnation_limit = 10 * dim

        while calls < self.budget:
            # Generate candidate by perturbing current best
            candidate = x + radius * self.rng.normal(0, 1, size=dim)
            candidate = np.clip(candidate, lb, ub)
            c_val = func(candidate)
            calls += 1

            if c_val < best_val:
                best_val = c_val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                x = candidate.copy()
                radius *= 1.2
                stagnation = 0
            else:
                radius *= 0.9
                stagnation += 1

            if stagnation > stagnation_limit and calls < self.budget:
                # Restart: new random point
                new_x = self.rng.uniform(lb, ub, size=dim)
                new_val = func(new_x)
                calls += 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                x = new_x.copy()
                radius = 0.2 * (ub - lb).max()
                stagnation = 0

        return best_val, best_x