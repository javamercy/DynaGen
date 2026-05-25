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
        # initial point
        best_x = lb + (ub - lb) * self.rng.rand(dim)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1
        # initial step sizes per coordinate
        step = 0.1 * (ub - lb)
        # restart loop
        while evals < self.budget:
            # choose a random coordinate and direction
            i = self.rng.randint(dim)
            sign = 1 if self.rng.rand() < 0.5 else -1
            candidate = best_x + sign * step * np.eye(dim)[i]
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = candidate.copy()
                step[i] *= 1.2
                report_best(best_f, best_x)
            else:
                step[i] *= 0.5
            # restart if steps are too small
            if np.all(step < 1e-15 * (ub - lb)):
                best_x = lb + (ub - lb) * self.rng.rand(dim)
                best_f = func(best_x)
                evals += 1
                report_best(best_f, best_x)
                step = 0.1 * (ub - lb)
        return best_f, best_x