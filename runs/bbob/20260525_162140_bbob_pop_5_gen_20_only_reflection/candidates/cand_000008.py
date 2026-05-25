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

        # Initialization
        x0 = lb + rng.rand(dim) * (ub - lb)
        f0 = func(x0)
        best_x = x0
        best_f = f0
        report_best(best_f, best_x)
        evals = 1

        # Parameters
        init_sigma = 0.2 * np.mean(ub - lb)
        sigma = init_sigma
        stagnation_limit = max(10, dim * 10)
        no_improve_steps = 0
        success_window = 5
        success_count = 0

        while evals < budget:
            # Gaussian sampling around best
            candidate = best_x + rng.normal(0, sigma, dim)
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1

            if f < best_f - 1e-12:
                best_f = f
                best_x = candidate
                report_best(best_f, best_x)
                no_improve_steps = 0
                success_count += 1
            else:
                no_improve_steps += 1

            # Adapt sigma
            if evals % success_window == 0:
                if success_count > 0.5 * success_window:
                    sigma = min(sigma * 1.2, 0.5 * np.mean(ub - lb))
                else:
                    sigma = max(sigma * 0.8, 0.01 * np.mean(ub - lb))
                success_count = 0

            # Restart if stagnation
            if no_improve_steps >= stagnation_limit:
                sigma = init_sigma
                no_improve_steps = 0

        return best_f, best_x