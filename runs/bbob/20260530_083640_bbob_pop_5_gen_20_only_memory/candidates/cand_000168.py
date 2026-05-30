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

        # Initial point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)

        # Simulated annealing parameters
        current_x = best_x.copy()
        current_f = best_f
        T0 = 1.0
        T = T0
        cooling = 0.99
        step0 = 0.2 * (ub - lb)
        step = step0.copy()
        min_step = 1e-10 * (ub - lb)

        iteration = 0
        while evals < budget:
            # Generate trial by adding Gaussian noise
            trial = current_x + step * rng.randn(dim)
            trial = np.clip(trial, lb, ub)
            trial_f = func(trial)
            evals += 1

            # Acceptance criterion
            delta = trial_f - current_f
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_x = trial
                current_f = trial_f
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial.copy()
                    report_best(best_f, best_x)

            # Update temperature and step size
            iteration += 1
            T = T0 * (cooling ** iteration)
            step = step0 * (T / T0) + min_step  # step decays with temperature

        return best_f, best_x