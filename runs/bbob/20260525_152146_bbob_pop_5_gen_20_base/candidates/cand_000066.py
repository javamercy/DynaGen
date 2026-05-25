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

        # Initialization
        best_x = rng.uniform(lb, ub, size=dim)
        best_f = func(best_x)
        budget -= 1
        report_best(best_f, best_x)

        # Initial step size
        step = np.mean(ub - lb) * 0.2
        min_step = 1e-8 * np.mean(ub - lb)
        max_step = np.mean(ub - lb)

        while budget > 0:
            # Generate trial
            trial = best_x + step * rng.randn(dim)
            trial = np.clip(trial, lb, ub)
            trial_f = func(trial)
            budget -= 1

            if trial_f < best_f:
                best_x = trial
                best_f = trial_f
                report_best(best_f, best_x)
                step *= 1.2  # Increase step after success
            else:
                step *= 0.8  # Decrease step after failure

            # Clamp step size to prevent extreme values
            step = np.clip(step, min_step, max_step)

        return best_f, best_x