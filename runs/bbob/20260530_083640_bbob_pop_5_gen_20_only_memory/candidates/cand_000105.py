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
        budget = self.budget
        rng = self.rng

        x = rng.uniform(lb, ub)
        cur_x = x.copy()
        cur_val = func(cur_x)
        evals = 1
        best_x = cur_x.copy()
        best_val = cur_val
        report_best(best_val, best_x)

        T0 = 1.0
        alpha = 0.99
        step0 = 0.1 * (ub - lb)
        iteration = 0

        while evals < budget:
            temperature = T0 * (alpha ** iteration)
            step = step0 * (1.0 - iteration / budget)
            dx = rng.normal(0, 1, dim) * step
            x_new = cur_x + dx
            x_new = np.clip(x_new, lb, ub)
            val_new = func(x_new)
            evals += 1

            if val_new < cur_val:
                cur_x = x_new
                cur_val = val_new
                if cur_val < best_val:
                    best_val = cur_val
                    best_x = cur_x.copy()
                    report_best(best_val, best_x)
            else:
                delta = val_new - cur_val
                if rng.rand() < np.exp(-delta / temperature):
                    cur_x = x_new
                    cur_val = val_new

            iteration += 1
            if evals >= budget:
                break

        return best_val, best_x