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
        best_x = None
        best_f = float('inf')
        calls = 0

        x = self.rng.uniform(lb, ub, self.dim)
        f = func(x)
        calls += 1
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)

        step = 0.1 * (ub - lb)

        while calls < self.budget:
            if self.rng.rand() < 0.3:
                x = self.rng.uniform(lb, ub, self.dim)
            else:
                factor = max(0.01, 1.0 - calls / self.budget)
                perturbation = self.rng.normal(0, step * factor, self.dim)
                x = best_x + perturbation
                x = np.clip(x, lb, ub)

            f = func(x)
            calls += 1

            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
                step *= 0.95

            if calls >= self.budget:
                break

        return best_f, best_x