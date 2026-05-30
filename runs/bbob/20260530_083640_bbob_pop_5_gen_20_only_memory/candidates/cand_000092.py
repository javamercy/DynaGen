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
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_x = lb + (ub - lb) * rng.rand(dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1
        last_improvement = 0

        for i in range(1, budget):
            if i - last_improvement > max(1, budget // 10):
                x = lb + (ub - lb) * rng.rand(dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                    last_improvement = i
                continue

            frac = 1.0 - (i - 1) / (budget - 1) * 0.9
            step_scale = frac * 0.1 * (ub - lb)
            candidate = best_x + rng.randn(dim) * step_scale
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                last_improvement = i

        return best_val, best_x