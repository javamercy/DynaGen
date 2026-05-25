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
        # initial point uniform random within bounds
        x = self.rng.uniform(lb, ub)
        best_x = x.copy()
        best_val = func(x)
        calls = 1
        report_best(best_val, best_x)
        step = 0.1 * (ub - lb)  # initial step sizes per dimension
        min_step = 1e-10 * (ub - lb)
        val = best_val
        while calls < self.budget:
            improved = False
            for i in range(dim):
                if calls >= self.budget:
                    break
                # positive direction
                x_new = x.copy()
                x_new[i] += step[i]
                x_new = np.clip(x_new, lb, ub)
                val_new = func(x_new)
                calls += 1
                if val_new < val:
                    x = x_new
                    val = val_new
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                    improved = True
                    break
                # negative direction
                x_new = x.copy()
                x_new[i] -= step[i]
                x_new = np.clip(x_new, lb, ub)
                val_new = func(x_new)
                calls += 1
                if val_new < val:
                    x = x_new
                    val = val_new
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                    improved = True
                    break
            if not improved:
                step *= 0.5
            else:
                step *= 1.1  # slight increase
            step = np.maximum(step, min_step)
            # occasional random direction probe
            if calls < self.budget and self.rng.uniform() < 0.1:
                d = self.rng.normal(size=dim)
                d = d / (np.linalg.norm(d) + 1e-10)
                x_new = x + step * d
                x_new = np.clip(x_new, lb, ub)
                val_new = func(x_new)
                calls += 1
                if val_new < val:
                    x = x_new
                    val = val_new
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
        return best_val, best_x