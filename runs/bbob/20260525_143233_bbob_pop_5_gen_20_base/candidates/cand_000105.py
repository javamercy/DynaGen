import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.restart_threshold = max(20, 2 * dim)
        self.T0 = 1.0
        self.cooling_rate = 0.95

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Evaluate initial point
        best_x = rng.uniform(lb, ub, dim).astype(np.float64)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        if budget == 1:
            return best_val, best_x

        # Initialize current point
        x = best_x.copy()
        f = best_val
        T = self.T0
        step_size = 0.2 * (ub - lb)  # per-coordinate step size
        no_improve = 0

        while evals < budget:
            # Generate neighbor
            perturbation = rng.normal(0, 1, dim) * step_size * T / self.T0
            x_new = x + perturbation
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            evals += 1

            # Acceptance criterion
            if f_new < f:
                x = x_new
                f = f_new
                if f_new < best_val:
                    best_val = f_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                no_improve = 0
            else:
                delta = f_new - f
                if rng.rand() < np.exp(-delta / T):
                    x = x_new
                    f = f_new
                no_improve += 1

            # Cool temperature
            T = self.T0 * (self.cooling_rate ** evals)

            # Restart if no improvement
            if no_improve >= self.restart_threshold and evals < budget:
                # Reset to best point with a small perturbation or random
                x = best_x + rng.normal(0, 1, dim) * step_size * 0.1
                x = np.clip(x, lb, ub)
                f = func(x)
                evals += 1
                if f < best_val:
                    best_val = f
                    best_x = x.copy()
                    report_best(best_val, best_x)
                T = self.T0
                no_improve = 0

        return best_val, best_x