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
        budget = self.budget
        rng = self.rng

        # Initial random point
        best_x = lb + (ub - lb) * rng.rand(self.dim)
        best_val = func(best_x)
        evals += 1
        self.best_val = best_val
        self.best_x = best_x.copy()
        report_best(best_val, best_x)

        # Step size initialization
        sigma = (ub - lb).mean() / 4.0
        min_sigma = 1e-10 * (ub - lb).mean()

        while evals < budget:
            # Generate offspring
            noise = rng.randn(self.dim)
            y = best_x + sigma * noise
            y = np.clip(y, lb, ub)
            y_val = func(y)
            evals += 1

            if y_val < best_val:
                best_val = y_val
                best_x = y.copy()
                report_best(best_val, best_x)
                sigma *= 1.2  # Increase step size on success
            else:
                sigma *= 0.8  # Decrease on failure

            # Avoid step size becoming too small
            if sigma < min_sigma:
                sigma = (ub - lb).mean() / 2.0

        return best_val, best_x