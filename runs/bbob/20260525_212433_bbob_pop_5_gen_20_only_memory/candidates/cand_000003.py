import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget

        # initial point
        best_x = lb + self.rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        radius = 0.2 * (ub - lb)
        patience = max(5, dim)
        no_improve = 0

        while calls < budget:
            # sample around best with adaptive radius
            candidate = best_x + radius * self.rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1

            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                radius *= 1.2
                no_improve = 0
            else:
                radius *= 0.8
                no_improve += 1

            # restart if stagnation
            if no_improve >= patience and calls < budget:
                best_x = lb + self.rng.rand(dim) * (ub - lb)
                best_val = func(best_x)
                calls += 1
                report_best(best_val, best_x)
                radius = 0.2 * (ub - lb)
                no_improve = 0

        return best_val, best_x