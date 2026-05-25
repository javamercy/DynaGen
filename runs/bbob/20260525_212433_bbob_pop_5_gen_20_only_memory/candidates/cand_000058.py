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
        x_curr = lb + (ub - lb) * rng.rand(self.dim)
        f_curr = func(x_curr)
        evals += 1
        self.best_val = f_curr
        self.best_x = x_curr.copy()
        report_best(self.best_val, self.best_x)

        if budget == 1:
            return self.best_val, self.best_x

        # Simulated annealing parameters
        T0 = 1.0
        cooling_rate = 0.99
        sigma = (ub - lb) * 0.2  # step size
        T = T0

        while evals < budget:
            # Generate candidate
            candidate = x_curr + sigma * rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1

            delta = f_candidate - f_curr
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                x_curr = candidate
                f_curr = f_candidate
                if f_candidate < self.best_val:
                    self.best_val = f_candidate
                    self.best_x = candidate.copy()
                    report_best(self.best_val, self.best_x)

            # Cool down
            T *= cooling_rate

        return self.best_val, self.best_x