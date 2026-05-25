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

        best_x = rng.uniform(lb, ub, size=dim)
        best_f = func(best_x)
        budget -= 1
        report_best(best_f, best_x)

        sigma = 0.2 * (ub - lb).mean()  # initial step size
        min_sigma = 1e-12
        tau = 1.0 / np.sqrt(dim)
        stagnation_limit = max(1, int(0.01 * self.budget))  # restart after this many failed attempts
        no_improve_count = 0

        while budget > 0:
            candidate = best_x + sigma * rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_f = func(candidate)
            budget -= 1

            if candidate_f < best_f:
                best_x = candidate.copy()
                best_f = candidate_f
                report_best(best_f, best_x)
                sigma *= np.exp(tau)  # increase step size on success
                no_improve_count = 0
            else:
                sigma *= np.exp(-tau)  # decrease step size on failure
                no_improve_count += 1

            # Ensure sigma doesn't become too small
            if sigma < min_sigma:
                sigma = min_sigma

            # Restart if stuck
            if no_improve_count >= stagnation_limit and budget > 0:
                new_x = rng.uniform(lb, ub, size=dim)
                new_f = func(new_x)
                budget -= 1
                if new_f < best_f:
                    best_x = new_x.copy()
                    best_f = new_f
                    report_best(best_f, best_x)
                sigma = 0.2 * (ub - lb).mean()
                no_improve_count = 0

        return best_f, best_x