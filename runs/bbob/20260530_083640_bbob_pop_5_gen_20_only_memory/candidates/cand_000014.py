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
        best_x = lb + self.rng.rand(dim) * (ub - lb)
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)
        step = 0.2 * (ub - lb)
        evals_since_improvement = 0
        while evals < self.budget:
            improved = False
            order = self.rng.permutation(dim)
            for i in order:
                if evals >= self.budget:
                    break
                # Positive direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                evals_since_improvement += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    improved = True
                    evals_since_improvement = 0
                    break
                # Negative direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                f = func(trial)
                evals += 1
                evals_since_improvement += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step[i] = min(step[i] * 2, ub[i] - lb[i])
                    improved = True
                    evals_since_improvement = 0
                    break
                else:
                    step[i] = max(step[i] * 0.5, (ub[i] - lb[i]) * 1e-10)
            if not improved and evals < self.budget:
                # Random direction
                direction = self.rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + step * direction, lb, ub)
                f = func(trial)
                evals += 1
                evals_since_improvement += 1
                if f < best_f:
                    best_f = f
                    best_x = trial
                    report_best(best_f, best_x)
                    step = np.minimum(step * 2, ub - lb)
                    improved = True
                    evals_since_improvement = 0
            # Restart if stuck
            if evals_since_improvement >= dim * 2 and evals < self.budget:
                new_x = lb + self.rng.rand(dim) * (ub - lb)
                f_new = func(new_x)
                evals += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = new_x
                    report_best(best_f, best_x)
                step = 0.2 * (ub - lb)
                evals_since_improvement = 0
        return best_f, best_x