import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_y = np.inf
        self.n_calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial LHS
        n_initial = min(self.budget, max(self.dim * 2, self.budget // 2))
        if n_initial < 1:
            n_initial = 1
        points = self._lhs(lb, ub, n_initial)
        for x in points:
            if self.n_calls >= self.budget:
                break
            y = func(x)
            self.n_calls += 1
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()
                report_best(y, x)
        # Local search
        step_size = (ub - lb) * 0.1
        while self.n_calls < self.budget:
            dx = self.rng.normal(0, step_size, size=self.dim)
            x_new = self.best_x + dx
            x_new = np.clip(x_new, lb, ub)
            y = func(x_new)
            self.n_calls += 1
            if y < self.best_y:
                self.best_y = y
                self.best_x = x_new.copy()
                report_best(y, x_new)
            step_size *= 0.99
        return self.best_y, self.best_x

    def _lhs(self, lb, ub, n):
        points = np.zeros((n, self.dim))
        for i in range(self.dim):
            strata = np.linspace(lb[i], ub[i], n+1)[:-1]
            order = self.rng.permutation(n)
            points[:, i] = strata[order] + self.rng.uniform(0, (ub[i]-lb[i])/n, size=n)
        return points