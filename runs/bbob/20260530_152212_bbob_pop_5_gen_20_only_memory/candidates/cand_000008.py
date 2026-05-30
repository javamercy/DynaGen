import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.evals = 0
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # initial random sampling (10% of budget, at least 1)
        n_init = max(1, int(self.budget * 0.1))
        best_x = None
        best_f = None
        for _ in range(min(n_init, self.budget)):
            x = np.random.uniform(lb, ub, size=self.dim)
            f = func(x)
            self.evals += 1
            if best_f is None or f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        
        remaining = self.budget - self.evals
        # initial perturbation scale (smaller than parent)
        sigma0 = 0.1 * (ub - lb).mean()
        for t in range(remaining):
            sigma = sigma0 * (1 - self.evals / self.budget) ** 1.0
            # low restart probability for exploitation
            if np.random.rand() < 0.05:
                x_try = np.random.uniform(lb, ub, size=self.dim)
            else:
                x_try = best_x + np.random.normal(0, sigma, size=self.dim)
                x_try = np.clip(x_try, lb, ub)
            f_try = func(x_try)
            self.evals += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                report_best(best_f, best_x)
        return best_f, best_x