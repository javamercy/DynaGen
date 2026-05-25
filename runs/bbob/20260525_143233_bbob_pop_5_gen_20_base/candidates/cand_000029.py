import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial point
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        if budget == 1:
            return best_val, best_x

        # Current state
        x_cur = best_x.copy()
        f_cur = best_val

        # Temperature schedule
        T0 = 1.0
        T_min = 1e-3
        cooling_factor = np.exp(np.log(T_min / T0) / (budget - 1))
        sigma_factor = 0.2  # step size factor relative to domain range
        T = T0

        while evals < budget:
            # Propose new point
            sigma = sigma_factor * (ub - lb) * (T / T0)
            noise = np.random.normal(0, sigma, dim)
            x_new = x_cur + noise
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            evals += 1

            # Update best
            if f_new < best_val:
                best_val = f_new
                best_x = x_new.copy()
                report_best(best_val, best_x)

            # Metropolis acceptance criterion
            delta = f_new - f_cur
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                x_cur = x_new
                f_cur = f_new

            # Cool down
            T *= cooling_factor

        return best_val, best_x