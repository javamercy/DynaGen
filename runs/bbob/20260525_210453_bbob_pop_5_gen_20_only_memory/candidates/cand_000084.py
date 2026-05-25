import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial random point
        x = np.random.uniform(lb, ub)
        val = func(x)
        fcalls = 1
        best_x = x.copy()
        best_f = val
        report_best(best_f, best_x)
        # SA parameters
        T0 = 1.0
        sigma0 = 0.2 * (ub - lb)  # step size per dimension
        while fcalls < budget:
            frac = fcalls / budget
            T = max(T0 * (1 - frac), 1e-10)  # linear cooling, avoid zero
            # Propose new point
            step = sigma0 * (T / T0) * np.random.randn(dim)
            x_new = np.clip(x + step, lb, ub)
            val_new = func(x_new)
            fcalls += 1
            # Acceptance criterion
            delta = val_new - val
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x = x_new
                val = val_new
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    report_best(best_f, best_x)
        return best_f, best_x