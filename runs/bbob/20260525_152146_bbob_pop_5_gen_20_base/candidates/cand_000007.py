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
        initial_radius = 0.05 * (ub - lb)
        radius = initial_radius.copy()
        stagnation_limit = 10
        stagnation_count = 0
        expand_factor = 1.1
        shrink_factor = 0.9
        pool_size = min(3, dim)
        while calls < self.budget:
            if calls >= self.budget:
                break
            actual_pool = min(pool_size, self.budget - calls)
            if actual_pool <= 0:
                break
            candidates = best_x + np.random.uniform(-1, 1, (actual_pool, dim)) * radius
            candidates = np.clip(candidates, lb, ub)
            values = np.array([func(c) for c in candidates])
            calls += actual_pool
            min_idx = np.argmin(values)
            cand_best_val = values[min_idx]
            if cand_best_val < best_val:
                best_val = cand_best_val
                best_x = candidates[min_idx]
                report_best(best_val, best_x)
                radius = np.minimum(radius * expand_factor, (ub - lb) * 0.25)
                stagnation_count = 0
            else:
                stagnation_count += 1
                radius = radius * shrink_factor
            if stagnation_count >= stagnation_limit:
                if calls >= self.budget:
                    break
                candidate = np.random.uniform(lb, ub)
                val = func(candidate)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate
                    report_best(best_val, best_x)
                radius = initial_radius.copy()
                stagnation_count = 0
        return best_val, best_x