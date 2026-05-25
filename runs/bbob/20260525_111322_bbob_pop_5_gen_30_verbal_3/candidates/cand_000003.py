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
        # initial random point
        x = lb + (ub - lb) * self.rng.rand(dim)
        val = func(x)
        evals = 1
        best_x = x.copy()
        best_val = val
        report_best(best_val, best_x)
        # adaptive local search
        radius = 0.2  # relative to range
        stagnation = 0
        max_stagnation = int(self.budget * 0.15)
        if max_stagnation < 5:
            max_stagnation = 5
        while evals < self.budget:
            # sample candidate around best
            range_vec = ub - lb
            step = radius * range_vec
            x_cand = best_x + self.rng.uniform(-step, step)
            x_cand = np.clip(x_cand, lb, ub)
            val_cand = func(x_cand)
            evals += 1
            if val_cand < best_val:
                best_val = val_cand
                best_x = x_cand.copy()
                report_best(best_val, best_x)
                radius *= 1.2
                stagnation = 0
            else:
                stagnation += 1
                radius *= 0.95
                if radius < 1e-10:
                    radius = 0.01
                if stagnation >= max_stagnation:
                    # restart
                    x_new = lb + (ub - lb) * self.rng.rand(dim)
                    if evals < self.budget:
                        val_new = func(x_new)
                        evals += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        radius = 0.2
                        stagnation = 0
                    else:
                        break
            if radius > 0.5:
                radius = 0.5
        return best_val, best_x