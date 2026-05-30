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

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        current_x = best_x.copy()
        current_val = best_val

        max_iter = budget - 1
        if max_iter <= 0:
            return best_val, best_x

        # Linear cooling schedules
        T0 = 1.0
        T_end = 0.001
        step0 = 0.1 * (ub - lb)
        step_end = 0.001 * (ub - lb)

        for i in range(max_iter):
            t = i / max_iter
            T = T0 * (1 - t) + T_end * t
            step = step0 * (1 - t) + step_end * t

            # Generate perturbation
            candidate = current_x + step * rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1

            delta = val - current_val
            if delta < 0:
                current_x = candidate
                current_val = val
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
            else:
                if rng.rand() < np.exp(-delta / T):
                    current_x = candidate
                    current_val = val

            if calls >= budget:
                break

        return best_val, best_x