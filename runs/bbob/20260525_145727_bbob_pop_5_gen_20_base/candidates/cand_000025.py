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

        # initial point
        x = rng.uniform(lb, ub)
        val = func(x)
        evals = 1
        best_x = x.copy()
        best_val = val
        report_best(best_val, best_x)

        # temperature schedule: T0=1, final=0.01
        alpha = 0.01 ** (1.0 / budget)
        T = 1.0

        # initial step size per dimension
        step_size = 0.2 * (ub - lb)

        # acceptance rate tracking
        window = min(50, budget)
        accept_count = 0
        window_count = 0

        while evals < budget:
            # generate candidate
            perturbation = rng.randn(dim) * step_size
            cand = np.clip(x + perturbation, lb, ub)
            cand_val = func(cand)
            evals += 1

            # metropolis acceptance
            delta = cand_val - val
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                x = cand
                val = cand_val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                accept_count += 1

            # update acceptance rate window
            window_count += 1
            if window_count >= window:
                rate = accept_count / window
                if rate > 0.5:
                    step_size *= 1.1
                else:
                    step_size *= 0.9
                # ensure step size not too small or large
                step_size = np.clip(step_size, 1e-8 * (ub - lb), 0.5 * (ub - lb))
                accept_count = 0
                window_count = 0

            # cool temperature
            T *= alpha

        return best_val, best_x