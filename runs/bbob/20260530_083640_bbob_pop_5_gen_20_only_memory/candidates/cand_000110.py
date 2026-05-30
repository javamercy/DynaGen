import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        if self.budget <= 0:
            return None, None
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        budget = self.budget
        evals = 0
        best_val = np.inf
        best_x = None

        # initial random points
        n_init = max(1, min(budget // 10, 10 * dim))
        for _ in range(n_init):
            x = lb + rng.rand(dim) * (ub - lb)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # adaptive step size per dimension
        sigma = 0.2 * (ub - lb)
        sigma = np.maximum(sigma, 1e-12)

        no_improve = 0
        max_no_improve = max(10, 10 * dim)
        while evals < budget:
            trial = best_x + sigma * rng.randn(dim)
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)
                sigma = np.minimum(sigma * 1.2, ub - lb)
                no_improve = 0
            else:
                sigma = np.maximum(sigma * 0.85, (ub - lb) * 1e-12)
                no_improve += 1

            if no_improve >= max_no_improve and evals < budget:
                # random restart
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                sigma = 0.2 * (ub - lb)
                no_improve = 0

        return best_val, best_x