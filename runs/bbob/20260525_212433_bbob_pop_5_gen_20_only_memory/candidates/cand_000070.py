import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        budget = self.budget
        rng = self.rng

        # Initial point
        x = lb + (ub - lb) * rng.rand(self.dim)
        val = func(x)
        evals += 1
        if val < self.best_val:
            self.best_val = val
            self.best_x = x.copy()
            report_best(self.best_val, self.best_x)

        # Simulated annealing parameters
        step_size = 0.2 * (ub - lb)
        T = 1.0
        cooling_rate = 0.99
        current = x
        current_val = val

        while evals < budget:
            # Generate candidate
            trial = current + step_size * rng.randn(self.dim)
            trial = np.clip(trial, lb, ub)
            trial_val = func(trial)
            evals += 1

            delta = trial_val - current_val
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current = trial
                current_val = trial_val
                if trial_val < self.best_val:
                    self.best_val = trial_val
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)

            # Cool down
            T *= cooling_rate

        return self.best_val, self.best_x