import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = np.inf
        self.best_x = None
        self.calls = 0

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        x0 = np.random.uniform(lb, ub)
        y0 = func(x0)
        self.calls += 1
        self.update_best(y0, x0, func)
        while self.calls < self.budget:
            # decide global vs local
            if np.random.rand() < 0.2 or self.calls < 3:
                x = np.random.uniform(lb, ub)
            else:
                remaining = self.budget - self.calls
                sigma = max(1e-3, (remaining / self.budget) ** 2 * (ub - lb).mean() / 5)
                x = self.best_x + np.random.randn(self.dim) * sigma
                x = np.clip(x, lb, ub)
            y = func(x)
            self.calls += 1
            self.update_best(y, x, func)
        return self.best_value, self.best_x

    def update_best(self, y, x, func):
        if y < self.best_value:
            self.best_value = y
            self.best_x = x.copy()
            report_best(y, x)