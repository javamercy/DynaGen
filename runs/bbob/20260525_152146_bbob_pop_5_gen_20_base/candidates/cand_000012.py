import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub

        def clip(x):
            return np.clip(x, lb, ub)

        # Initial point
        best_x = rng.uniform(lb, ub, size=dim)
        best_f = func(best_x)
        budget -= 1
        report_best(best_f, best_x)

        # Initial step size: 0.1 of domain range
        sigma0 = 0.1 * (ub - lb)
        max_budget = self.budget
        eval_count = 1

        while budget > 0:
            # Decreasing step size linearly
            frac = eval_count / max_budget
            sigma = sigma0 * (1 - frac) + 1e-8 * (ub - lb)  # avoid zero

            # Exploration vs exploitation
            if rng.rand() < 0.1:
                candidate = rng.uniform(lb, ub, size=dim)
            else:
                pert = rng.normal(0, sigma, size=dim)
                candidate = best_x + pert
                candidate = clip(candidate)

            cand_f = func(candidate)
            budget -= 1
            eval_count += 1

            if cand_f < best_f:
                best_f = cand_f
                best_x = candidate.copy()
                report_best(best_f, best_x)

        return best_f, best_x