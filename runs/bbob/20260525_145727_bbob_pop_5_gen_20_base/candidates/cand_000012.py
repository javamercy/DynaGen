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

        n_init = max(1, budget // 5)
        best_x = None
        best_y = np.inf

        for _ in range(n_init):
            x = rng.uniform(lb, ub, dim)
            y = func(x)
            if y < best_y:
                best_y = y
                best_x = x.copy()
                report_best(best_y, best_x)

        remaining = budget - n_init
        if remaining > 0:
            step_scale = 0.1 * (ub - lb)
            for i in range(remaining):
                frac = i / remaining
                step = step_scale * (1 - frac)
                trial = best_x + rng.randn(dim) * step
                trial = np.clip(trial, lb, ub)
                y = func(trial)
                if y < best_y:
                    best_y = y
                    best_x = trial.copy()
                    report_best(best_y, best_x)

        return best_y, best_x