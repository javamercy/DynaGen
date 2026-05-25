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
        best_x = lb + (ub - lb) * self.rng.rand(self.dim)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1
        step = 0.1 * np.mean(ub - lb)
        while evals < self.budget:
            improved = False
            # iterate over coordinate axes
            for i in range(self.dim):
                if evals >= self.budget:
                    break
                d = np.zeros(self.dim)
                d[i] = 1.0
                # positive step
                candidate = best_x + step * d
                candidate = np.clip(candidate, lb, ub)
                f_val = func(candidate)
                evals += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate.copy()
                    improved = True
                    report_best(best_f, best_x)
                    break
                if evals >= self.budget:
                    break
                # negative step
                candidate = best_x - step * d
                candidate = np.clip(candidate, lb, ub)
                f_val = func(candidate)
                evals += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate.copy()
                    improved = True
                    report_best(best_f, best_x)
                    break
            if improved:
                step *= 1.2
            else:
                step *= 0.5
                if step < 1e-15:
                    break
        return best_f, best_x