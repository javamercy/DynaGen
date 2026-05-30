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
        x0 = rng.uniform(lb, ub, size=dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        report_best(best_f, best_x)
        evals = 1

        # Adaptive radius parameters
        radius = 0.2 * (ub - lb)  # initial radius per dimension
        shrink_factor = 0.9
        expand_factor = 1.2
        stagnation_limit = max(50, int(budget * 0.05))  # restart after this many evals without improvement
        no_improve = 0

        while evals < budget:
            # Sample candidate around best with Gaussian perturbation
            candidate = best_x + rng.normal(0, radius, size=dim)
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1

            if f_candidate < best_f:
                best_f = f_candidate
                best_x = candidate.copy()
                report_best(best_f, best_x)
                no_improve = 0
                # Expand radius on improvement
                radius = np.minimum(radius * expand_factor, (ub - lb) * 0.5)
            else:
                no_improve += 1
                # Shrink radius on failure
                radius = radius * shrink_factor
                # Prevent radius from becoming too small
                radius = np.maximum(radius, 1e-8 * (ub - lb))

            # Restart if stagnation
            if no_improve >= stagnation_limit:
                x_restart = rng.uniform(lb, ub, size=dim)
                f_restart = func(x_restart)
                evals += 1
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                    report_best(best_f, best_x)
                # Reset radius
                radius = 0.2 * (ub - lb)
                no_improve = 0

        return best_f, best_x