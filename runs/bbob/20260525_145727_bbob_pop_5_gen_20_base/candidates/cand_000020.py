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
        rng = self.rng
        budget = self.budget

        ranges = ub - lb
        avg_range = np.mean(ranges)
        sigma = 0.2 * avg_range
        sigma_min = 1e-5 * avg_range
        sigma_max = 0.5 * avg_range

        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        lambda_ = 5  # offspring per generation

        while evals < budget:
            best_val_start = best_val
            successes = 0
            for _ in range(lambda_):
                if evals >= budget:
                    break
                trial = best_x + sigma * rng.randn(dim)
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                if val < best_val_start:
                    successes += 1
            # adapt sigma
            success_rate = successes / lambda_
            if success_rate > 0.2:
                sigma *= 1.2
            elif success_rate < 0.2:
                sigma /= 1.2
            sigma = np.clip(sigma, sigma_min, sigma_max)

        return best_val, best_x