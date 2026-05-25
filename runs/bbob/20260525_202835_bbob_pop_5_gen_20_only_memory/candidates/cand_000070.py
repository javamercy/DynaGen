import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.T0 = 0.5
        self.Tf = 0.01

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        x = self.rng.uniform(lb, ub)
        best_x = x.copy()
        best_val = func(x)
        evaluations = 1
        report_best(best_val, best_x)
        current_x = x.copy()
        current_val = best_val
        t = 0
        while evaluations < self.budget:
            t += 1
            T = self.T0 + (self.Tf - self.T0) * (t / self.budget)
            scale = (ub - lb) * 0.2 * (T / self.T0)
            new_x = current_x + self.rng.normal(0, scale)
            new_x = np.clip(new_x, lb, ub)
            new_val = func(new_x)
            evaluations += 1
            delta = new_val - current_val
            if delta < 0:
                current_x = new_x
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
            else:
                if self.rng.random() < np.exp(-delta / T):
                    current_x = new_x
                    current_val = new_val
        return best_val, best_x