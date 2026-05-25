import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.T0 = 1.0
        self.cooling = 0.99
        self.sigma0 = 0.2
        self.restart_threshold = max(1, int(0.1 * budget))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initialize
        x = rng.uniform(lb, ub, dim)
        fx = func(x)
        evals = 1
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)

        T = self.T0
        sigma = self.sigma0 * (ub - lb)
        no_improve = 0

        while evals < budget:
            # Propose new point
            y = x + sigma * rng.randn(dim)
            y = np.clip(y, lb, ub)
            fy = func(y)
            evals += 1

            # Accept or reject
            if fy < fx:
                x = y
                fx = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    report_best(best_f, best_x)
                no_improve = 0
            else:
                delta = fy - fx
                if rng.rand() < np.exp(-delta / max(T, 1e-10)):
                    x = y
                    fx = fy
                no_improve += 1

            # Cool temperature
            T *= self.cooling

            # Adapt sigma: simple rule based on acceptance
            # (implicitly through no_improve, but we keep sigma fixed for simplicity)

            # Restart if stagnation
            if no_improve >= self.restart_threshold:
                x = rng.uniform(lb, ub, dim)
                fx = func(x)
                evals += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    report_best(best_f, best_x)
                T = self.T0
                no_improve = 0

        return best_f, best_x