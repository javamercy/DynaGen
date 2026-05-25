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
        max_evals = self.budget
        # initial random point
        x = self.rng.uniform(lb, ub)
        val = func(x)
        best_x = x.copy()
        best_val = val
        report_best(best_val, best_x)
        evals = 1
        T0 = 1.0
        while evals < max_evals:
            t = T0 * np.exp(-3.0 * evals / max_evals)
            step = self.rng.normal(0, 0.2 * (ub - lb) * t, size=dim)
            x_new = np.clip(x + step, lb, ub)
            val_new = func(x_new)
            evals += 1
            delta = val_new - val
            if delta < 0 or self.rng.uniform() < np.exp(-delta / max(t, 1e-10)):
                x = x_new
                val = val_new
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
            if evals >= max_evals:
                break
        return best_val, best_x