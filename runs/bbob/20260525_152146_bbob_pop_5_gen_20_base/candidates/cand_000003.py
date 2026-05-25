import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        best_x = np.random.uniform(lb, ub)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)
        initial_radius = 0.2 * (ub - lb)
        radius = initial_radius.copy()
        stagnation_limit = max(5, int(self.budget * 0.01))
        stagnation_count = 0
        shrink_factor = 0.5
        expand_factor = 1.5
        while calls < self.budget:
            candidate = best_x + np.random.uniform(-1, 1, dim) * radius
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1
            if val < best_val:
                best_val = val
                best_x = candidate
                report_best(best_val, best_x)
                radius = np.minimum(radius * expand_factor, (ub - lb) * 0.5)
                stagnation_count = 0
            else:
                stagnation_count += 1
                radius = radius * shrink_factor
            if stagnation_count >= stagnation_limit:
                candidate = np.random.uniform(lb, ub)
                val = func(candidate)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate
                    report_best(best_val, best_x)
                radius = initial_radius.copy()
                stagnation_count = 0
            if calls >= self.budget:
                break
        return best_val, best_x