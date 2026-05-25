import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # initial point
        x = lb + (ub - lb) * self.rng.rand(self.dim)
        f = func(x)
        evals += 1
        if f < self.best_val:
            self.best_val = f
            self.best_x = x.copy()
            report_best(self.best_val, self.best_x)
        # step size as fraction of average bound range
        step_size = 0.1 * (ub - lb).mean()
        T0 = 1.0
        T_end = 0.001
        while evals < self.budget:
            # generate neighbor
            perturbation = self.rng.normal(0, step_size, self.dim)
            x_new = x + perturbation
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            evals += 1
            if f_new < self.best_val:
                self.best_val = f_new
                self.best_x = x_new.copy()
                report_best(self.best_val, self.best_x)
            # accept or reject
            delta = f_new - f
            if delta < 0 or self.rng.rand() < np.exp(-delta / max(T0, 1e-10)):
                x = x_new
                f = f_new
            # geometric cooling
            T = T0 * (T_end / T0) ** (evals / self.budget)
        return self.best_val, self.best_x