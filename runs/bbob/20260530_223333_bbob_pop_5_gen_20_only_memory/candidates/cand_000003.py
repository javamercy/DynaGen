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
        bounds_range = ub - lb
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial random point within bounds
        best_x = lb + rng.random(dim) * bounds_range
        best_val = func(best_x)
        calls = 1
        from . import report_best
        report_best(best_val, best_x)

        # Adaptive parameters
        radius = 0.2 * np.mean(bounds_range)
        stagnation_limit = max(5 * dim, 10)
        stagnation = 0

        while calls < budget:
            # Sample perturbation
            step = rng.normal(0, radius, size=dim)
            new_x = best_x + step
            new_x = np.clip(new_x, lb, ub)
            new_val = func(new_x)
            calls += 1

            if new_val < best_val:
                best_val = new_val
                best_x = new_x
                report_best(best_val, best_x)
                radius *= 1.1  # expand on success
                stagnation = 0
            else:
                radius *= 0.9  # shrink on failure
                stagnation += 1

            # Restart if stagnation
            if stagnation >= stagnation_limit:
                best_x = lb + rng.random(dim) * bounds_range
                best_val = func(best_x)
                calls += 1
                report_best(best_val, best_x)
                radius = 0.2 * np.mean(bounds_range)
                stagnation = 0

        return best_val, best_x