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
        # Initial point uniformly in bounds
        x = lb + (ub - lb) * self.rng.rand(self.dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        report_best(best_f, best_x)
        evals = 1

        # Step size relative to domain range
        sigma = 0.2 * np.mean(ub - lb)

        while evals < self.budget:
            # Generate candidate by adding Gaussian noise
            y = x + sigma * self.rng.randn(self.dim)
            y = np.clip(y, lb, ub)
            f_y = func(y)
            evals += 1
            if f_y < f_x:  # improvement over current point
                x = y
                f_x = f_y
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
                    report_best(best_f, best_x)
                sigma *= 1.2  # increase step
            else:
                sigma *= 0.8  # decrease step
            # Prevent sigma from becoming too small
            min_sigma = 1e-15 * np.mean(ub - lb)
            if sigma < min_sigma:
                sigma = 0.1 * np.mean(ub - lb)
        return best_f, best_x