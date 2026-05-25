import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.T0 = 1.0
        self.T_end = 1e-8
        self.alpha = (self.T_end / self.T0) ** (1.0 / budget)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        x = self.rng.uniform(lb, ub)
        val = func(x)
        best_val = val
        best_x = x.copy()
        evaluations = 1
        report_best(best_val, best_x)
        initial_sigma = 0.1 * (ub - lb)
        current_x = x.copy()
        current_val = val
        while evaluations < self.budget:
            T = self.T0 * (self.alpha ** evaluations)
            sigma = initial_sigma * T / self.T0
            new_x = current_x + self.rng.normal(0, sigma, size=dim)
            new_x = np.clip(new_x, lb, ub)
            new_val = func(new_x)
            evaluations += 1
            if new_val < best_val:
                best_val = new_val
                best_x = new_x.copy()
                report_best(best_val, best_x)
            if new_val < current_val:
                current_x = new_x.copy()
                current_val = new_val
            else:
                acceptance = np.exp((current_val - new_val) / T)
                if self.rng.random() < acceptance:
                    current_x = new_x.copy()
                    current_val = new_val
        return best_val, best_x