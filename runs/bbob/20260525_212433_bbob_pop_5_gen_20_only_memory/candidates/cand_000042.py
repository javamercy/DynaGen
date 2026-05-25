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
        rng = self.rng
        dim = self.dim

        # Initial random point
        x = lb + (ub - lb) * rng.rand(dim)
        fx = func(x)
        evals += 1
        if fx < self.best_val:
            self.best_val = fx
            self.best_x = x.copy()
            report_best(self.best_val, self.best_x)

        # If budget exhausted, return
        if evals >= self.budget:
            return self.best_val, self.best_x

        # Initialize step size as a fraction of the domain range
        sigma = 0.2 * (ub - lb)  # vector of per-dimension step sizes
        # We'll use a common scalar step size for simplicity
        sigma_scalar = np.mean(sigma)
        min_sigma = 1e-12 * np.mean(ub - lb)
        max_sigma = np.mean(ub - lb)

        # Main loop
        while evals < self.budget:
            # Generate offspring via Gaussian mutation
            noise = rng.randn(dim)
            y = x + sigma_scalar * noise * (ub - lb)  # scale by domain width
            y = np.clip(y, lb, ub)
            fy = func(y)
            evals += 1
            if fy < self.best_val:
                self.best_val = fy
                self.best_x = y.copy()
                report_best(self.best_val, self.best_x)

            # Selection: if improvement, replace parent and increase sigma
            if fy < fx:
                x = y
                fx = fy
                sigma_scalar *= 1.2  # increase step size
            else:
                sigma_scalar *= 0.85  # decrease step size

            # Clamp sigma to avoid extremes
            sigma_scalar = np.clip(sigma_scalar, min_sigma, max_sigma)

        return self.best_val, self.best_x