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

        # Initial point
        current_x = rng.uniform(lb, ub)
        current_f = func(current_x)
        budget -= 1
        best_x = current_x.copy()
        best_f = current_f
        report_best(best_f, best_x)

        # Temperature schedule
        T0 = 1.0
        T_end = 1e-5
        alpha = (T_end / T0) ** (1.0 / budget) if budget > 0 else 1.0
        T = T0

        # Step size (relative to bounds)
        step_size = 0.2 * (ub - lb)

        while budget > 0:
            # Generate candidate by adding Gaussian noise
            noise = rng.normal(0, step_size, size=dim)
            candidate = current_x + noise
            candidate = np.clip(candidate, lb, ub)
            candidate_f = func(candidate)
            budget -= 1

            delta = candidate_f - current_f
            if delta < 0:
                # Accept improvement
                current_x = candidate
                current_f = candidate_f
                if candidate_f < best_f:
                    best_x = candidate.copy()
                    best_f = candidate_f
                    report_best(best_f, best_x)
            else:
                # Accept with probability
                if rng.rand() < np.exp(-delta / T):
                    current_x = candidate
                    current_f = candidate_f

            # Cool down
            T *= alpha

        return best_f, best_x