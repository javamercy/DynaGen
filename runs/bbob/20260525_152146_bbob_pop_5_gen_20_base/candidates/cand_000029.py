import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        def clip(x):
            return np.clip(x, lb, ub)

        # Initialize with a random point
        x = rng.uniform(lb, ub, size=dim)
        f = func(x)
        budget -= 1
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)

        if budget == 0:
            return best_f, best_x

        # Simulated Annealing parameters
        T = 100.0  # initial temperature
        alpha = 0.995  # cooling factor
        sigma = 0.1 * (ub - lb)  # step size

        while budget > 0:
            # generate candidate
            pert = rng.normal(0, sigma, size=dim)
            x_new = clip(x + pert)
            f_new = func(x_new)
            budget -= 1

            # accept based on Metropolis criterion
            delta = f_new - f
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                x = x_new
                f = f_new
                if f < best_f:
                    best_x = x.copy()
                    best_f = f
                    report_best(best_f, best_x)

            # cool down
            T *= alpha

        return best_f, best_x