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
        dim = self.dim
        rng = self.rng
        budget = self.budget
        evals = 0
        # Initial point
        x = lb + (ub - lb) * rng.rand(dim)
        fx = func(x)
        evals += 1
        if fx < self.best_val:
            self.best_val = fx
            self.best_x = x.copy()
            report_best(self.best_val, self.best_x)
        current_x = x
        current_fx = fx
        # Temperature schedule
        T0 = 1.0
        T = T0
        # Geometric cooling factor to reach T_end=1e-3 at budget
        remaining = max(budget - evals, 1)
        alpha = (1e-3 / T0) ** (1.0 / remaining)
        # Main loop
        while evals < budget:
            step = 0.2 * (ub - lb) * (T / T0)
            y = current_x + step * rng.randn(dim)
            y = np.clip(y, lb, ub)
            fy = func(y)
            evals += 1
            if fy < self.best_val:
                self.best_val = fy
                self.best_x = y.copy()
                report_best(self.best_val, self.best_x)
            # Acceptance criterion
            if fy < current_fx:
                current_x = y
                current_fx = fy
            else:
                delta = fy - current_fx
                prob = np.exp(-delta / T) if T > 0 else 0.0
                if rng.rand() < prob:
                    current_x = y
                    current_fx = fy
            # Update temperature
            T *= alpha
            if T < 1e-30:
                T = 1e-30
        return self.best_val, self.best_x