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
        best_x = rng.uniform(lb, ub, size=dim)
        best_f = func(best_x)
        budget -= 1
        report_best(best_f, best_x)

        # Step size: 20% of average domain range
        sigma = 0.2 * np.mean(ub - lb)

        for _ in range(budget):
            trial = best_x + sigma * rng.randn(dim)
            trial = np.clip(trial, lb, ub)
            trial_f = func(trial)
            if trial_f < best_f:
                best_x = trial.copy()
                best_f = trial_f
                report_best(best_f, best_x)

        return best_f, best_x