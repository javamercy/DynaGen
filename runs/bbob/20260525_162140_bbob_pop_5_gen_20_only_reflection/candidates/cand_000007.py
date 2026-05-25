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
        best_x = self.rng.uniform(lb, ub)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1
        # step sizes as fraction of range
        step = 0.2 * (ub - lb)
        min_step = 1e-3 * (ub - lb)
        # cycle through coordinates
        coord_order = np.arange(dim)
        while evals < self.budget:
            for coord in coord_order:
                if evals >= self.budget:
                    break
                # try positive direction
                candidate = best_x.copy()
                candidate[coord] += step[coord]
                candidate = np.clip(candidate, lb, ub)
                f_candidate = func(candidate)
                evals += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
                    step[coord] *= 1.2
                    continue  # skip negative direction on success
                # try negative direction
                if evals >= self.budget:
                    step[coord] *= 0.5
                    break
                candidate2 = best_x.copy()
                candidate2[coord] -= step[coord]
                candidate2 = np.clip(candidate2, lb, ub)
                f_candidate2 = func(candidate2)
                evals += 1
                if f_candidate2 < best_f:
                    best_f = f_candidate2
                    best_x = candidate2.copy()
                    report_best(best_f, best_x)
                    step[coord] *= 1.2
                else:
                    step[coord] *= 0.5
                step[coord] = max(step[coord], min_step[coord])
        return best_f, best_x