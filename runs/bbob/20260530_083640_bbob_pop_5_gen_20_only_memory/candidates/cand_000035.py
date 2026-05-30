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
        diff = ub - lb

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        current_x = best_x.copy()
        current_val = best_val

        # Two sets of parameters for interpolation (exploratory vs exploitative)
        step0_1 = 0.2 * diff
        step_end_1 = 1e-4 * diff
        T_end_1 = 1e-3
        step0_2 = 0.1 * diff
        step_end_2 = 1e-6 * diff
        T_end_2 = 1e-4
        T0 = 1.0

        while calls < budget:
            t = (calls - 1) / (budget - 1) if budget > 1 else 1.0
            # Log-linear interpolation of parameters
            step0 = np.exp(np.log(step0_1) * (1 - t) + np.log(step0_2) * t)
            step_end = np.exp(np.log(step_end_1) * (1 - t) + np.log(step_end_2) * t)
            T_end = np.exp(np.log(T_end_1) * (1 - t) + np.log(T_end_2) * t)

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