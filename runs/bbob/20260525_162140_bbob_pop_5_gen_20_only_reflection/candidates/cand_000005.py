import numpy as np

def report_best(value, x):
    pass

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        best_f = np.inf
        best_x = None
        calls = 0

        # Phase 1: global exploration
        n_global = min(budget // 2, 50)
        for _ in range(n_global):
            x = lb + rng.uniform(0, 1, dim) * (ub - lb)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Phase 2: local refinement with occasional global jumps
        while calls < budget:
            if rng.uniform() < 0.2:
                x = lb + rng.uniform(0, 1, dim) * (ub - lb)
            else:
                sigma = 0.2 * (1 - calls / budget) * (ub - lb)
                x = best_x + rng.normal(0, sigma, dim)
                x = np.clip(x, lb, ub)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        return best_f, best_x