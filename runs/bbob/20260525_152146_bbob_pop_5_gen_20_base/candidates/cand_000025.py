import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # initial point
        x = rng.uniform(lb, ub)
        fx = func(x)
        remaining = self.budget - 1
        best_x = x.copy()
        best_f = fx
        report_best(best_f, best_x)

        # SA parameters
        T0 = 1.0
        T_min = 1e-3
        if remaining > 0:
            cooling = (T_min / T0) ** (1.0 / remaining)
        else:
            cooling = 1.0
        T = T0

        while remaining > 0:
            step = T * (ub - lb) * 0.1  # scale step size
            candidate = x + rng.normal(0, step)
            candidate = np.clip(candidate, lb, ub)
            fc = func(candidate)
            remaining -= 1
            # acceptance
            if fc < fx or rng.rand() < np.exp((fx - fc) / T):
                x = candidate
                fx = fc
                if fc < best_f:
                    best_x = candidate.copy()
                    best_f = fc
                    report_best(best_f, best_x)
            # cool
            T *= cooling

        return best_f, best_x