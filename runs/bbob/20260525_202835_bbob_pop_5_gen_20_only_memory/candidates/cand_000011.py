import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        # initial point
        x0 = lb + self.rng.random(dim) * (ub - lb)
        best_x = x0.copy()
        best_val = func(best_x)
        self.budget -= 1
        report_best(best_val, best_x)
        # global step size per coordinate
        step = 0.1 * (ub - lb)
        while self.budget > 0:
            improved = False
            perm = self.rng.permutation(dim)
            for i in perm:
                if self.budget <= 0:
                    break
                # try positive direction
                x_new = best_x.copy()
                x_new[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                val_new = func(x_new)
                self.budget -= 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                    improved = True
                    continue  # skip negative direction for this coordinate
                # try negative direction
                if self.budget <= 0:
                    break
                x_new[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                val_new = func(x_new)
                self.budget -= 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                    improved = True
            if not improved:
                step *= 0.5
        return best_val, best_x