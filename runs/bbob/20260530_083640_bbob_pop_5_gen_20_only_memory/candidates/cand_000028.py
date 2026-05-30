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
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        current_x = best_x.copy()
        current_val = best_val

        # Temperature and step size parameters (mutated)
        T0 = 1.0
        T_end = 1e-3  # less cooling
        step0 = 0.2 * (ub - lb)  # larger initial step
        step_end = 1e-4 * (ub - lb)

        while calls < budget:
            t = (calls - 1) / (budget - 1) if budget > 1 else 1.0
            T = T0 * (T_end / T0) ** t
            step = step0 * (step_end / step0) ** t

            candidate = current_x + step * rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1

            delta = val - current_val
            if delta < 0:
                current_x = candidate
                current_val = val
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
            else:
                if rng.rand() < np.exp(-delta / T):
                    current_x = candidate
                    current_val = val

        return best_val, best_x