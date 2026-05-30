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
        best_x = lb + self.rng.rand(self.dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)
        step = 0.2 * (ub - lb).mean()
        while evals < self.budget:
            improved = False
            perm = self.rng.permutation(self.dim)
            for i in perm:
                if evals >= self.budget:
                    break
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step, lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    improved = True
                    step *= 2.0
                    break
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] - step, lb[i], ub[i])
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    improved = True
                    step *= 2.0
                    break
            if improved:
                continue
            if evals < self.budget:
                direction = self.rng.randn(self.dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + step * direction, lb, ub)
                f = func(trial)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step *= 2.0
                    improved = True
            if not improved:
                step *= 0.5
        return best_f, best_x