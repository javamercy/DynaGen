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
        steps = 0.1 * np.ones(self.dim) * (ub - lb)
        no_improve_streak = 0
        max_streak = 2 * self.dim
        while evals < self.budget:
            if self.rng.rand() < 0.1 and evals < self.budget:
                candidate = lb + (ub - lb) * self.rng.rand(self.dim)
                f_val = func(candidate)
                evals += 1
                if f_val < best_f:
                    best_f = f_val
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    steps = 0.1 * (ub - lb)
                    no_improve_streak = 0
                    continue
                else:
                    no_improve_streak += 1
            else:
                dims = self.rng.permutation(self.dim)
                improved_this_loop = False
                for i in dims:
                    if evals >= self.budget:
                        break
                    candidate = best_x.copy()
                    candidate[i] += steps[i]
                    candidate[i] = np.clip(candidate[i], lb[i], ub[i])
                    f_val = func(candidate)
                    evals += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                        steps[i] *= 1.2
                        improved_this_loop = True
                        no_improve_streak = 0
                        break
                    else:
                        candidate = best_x.copy()
                        candidate[i] -= steps[i]
                        candidate[i] = np.clip(candidate[i], lb[i], ub[i])
                        f_val = func(candidate)
                        evals += 1
                        if f_val < best_f:
                            best_f = f_val
                            best_x = candidate.copy()
                            report_best(best_f, best_x)
                            steps[i] *= 1.2
                            improved_this_loop = True
                            no_improve_streak = 0
                            break
                        else:
                            steps[i] *= 0.5
                if not improved_this_loop:
                    no_improve_streak += 1
            if no_improve_streak >= max_streak or np.max(steps) < 1e-15:
                steps = 0.1 * (ub - lb)
                no_improve_streak = 0
                if evals < self.budget:
                    candidate = lb + (ub - lb) * self.rng.rand(self.dim)
                    f_val = func(candidate)
                    evals += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                        steps = 0.1 * (ub - lb)
        return best_f, best_x