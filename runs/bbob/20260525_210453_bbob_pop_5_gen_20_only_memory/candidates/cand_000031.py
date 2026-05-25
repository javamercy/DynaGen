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
        best_x = None
        best_f = np.inf
        fcalls = 0

        # initial random point
        x0 = np.random.uniform(lb, ub, size=dim)
        f0 = func(x0)
        fcalls += 1
        best_x = x0.copy()
        best_f = f0
        report_best(best_f, best_x)

        current_x = x0.copy()
        current_f = f0

        # SA parameters
        T0 = 100.0
        Tf = 1e-5
        max_iter = budget - 1
        if max_iter <= 0:
            return best_f, best_x
        cooling = (Tf / T0) ** (1.0 / max_iter)
        T = T0

        step_range = ub - lb
        step_factor = 0.1

        for _ in range(max_iter):
            # propose new point
            step_size = step_factor * step_range * (T / T0)
            new_x = current_x + step_size * np.random.randn(dim)
            new_x = np.clip(new_x, lb, ub)
            fnew = func(new_x)
            fcalls += 1

            # acceptance criterion
            if fnew < current_f or np.random.random() < np.exp((current_f - fnew) / T):
                current_x = new_x
                current_f = fnew
                if fnew < best_f:
                    best_f = fnew
                    best_x = new_x.copy()
                    report_best(best_f, best_x)

            # cooling
            T *= cooling

        return best_f, best_x