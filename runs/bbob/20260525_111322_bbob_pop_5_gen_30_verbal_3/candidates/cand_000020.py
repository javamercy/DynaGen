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
        step_sizes = np.full(self.dim, 0.1 * np.mean(ub - lb))
        expand = 1.2
        contract = 0.5
        while evals < self.budget:
            improved = False
            for i in range(self.dim):
                if evals >= self.budget:
                    break
                step = step_sizes[i]
                # positive direction
                candidate = best_x.copy()
                candidate[i] += step
                candidate = np.clip(candidate, lb, ub)
                f = func(candidate)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = candidate.copy()
                    improved = True
                    step_sizes[i] *= expand
                    report_best(best_f, best_x)
                    break
                # negative direction
                candidate = best_x.copy()
                candidate[i] -= step
                candidate = np.clip(candidate, lb, ub)
                f = func(candidate)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = candidate.copy()
                    improved = True
                    step_sizes[i] *= expand
                    report_best(best_f, best_x)
                    break
            if not improved:
                step_sizes *= contract
                if np.max(step_sizes) < 1e-15 and evals < self.budget:
                    # restart from random point
                    candidate = lb + (ub - lb) * self.rng.rand(self.dim)
                    f = func(candidate)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                    step_sizes = np.full(self.dim, 0.1 * np.mean(ub - lb))
        return best_f, best_x