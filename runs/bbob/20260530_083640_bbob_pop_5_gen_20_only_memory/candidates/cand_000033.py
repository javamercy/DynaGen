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
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Temperature schedule: geometric cooling
        T0 = 1.0  # initial temperature, adjustable
        alpha = 0.99  # cooling factor
        # number of iterations based on remaining budget
        max_iter = budget - evals
        T = T0
        current_x = best_x.copy()
        current_val = best_val

        for _ in range(max_iter):
            if evals >= budget:
                break
            # Generate neighbor by Gaussian perturbation
            step = 0.1 * (ub - lb) * rng.randn(dim)  # scale by range
            candidate_x = current_x + step * T  # temperature scales step
            candidate_x = np.clip(candidate_x, lb, ub)
            candidate_val = func(candidate_x)
            evals += 1
            delta = candidate_val - current_val
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_x = candidate_x
                current_val = candidate_val
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_x = candidate_x.copy()
                    report_best(best_val, best_x)
            # Cool down
            T *= alpha
            if evals >= budget:
                break

        return best_val, best_x