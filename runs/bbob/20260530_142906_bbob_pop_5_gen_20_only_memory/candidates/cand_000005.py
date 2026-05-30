import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        n_init = min(10, budget // 2)
        for _ in range(n_init):
            x = lb + self.rng.uniform(size=dim) * (ub - lb)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        remaining = budget - calls
        local_factor = 0.5
        min_factor = 1e-3
        for i in range(remaining):
            if self.rng.uniform() < 0.3 or best_x is None:
                x = lb + self.rng.uniform(size=dim) * (ub - lb)
            else:
                factor = max(min_factor, local_factor * (1 - calls/budget))
                step = (ub - lb) * factor
                x = best_x + self.rng.normal(0, 1, size=dim) * step
                x = np.clip(x, lb, ub)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        return best_f, best_x