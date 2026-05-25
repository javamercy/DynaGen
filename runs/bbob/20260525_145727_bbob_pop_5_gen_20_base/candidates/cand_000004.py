import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        x = lb + (ub - lb) * self.rng.rand(self.dim)
        x = np.clip(x, lb, ub)
        best_x = x.copy()
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        domain_range = ub - lb
        step_size = 0.1 * domain_range.mean()
        min_step = 1e-8 * domain_range.mean()
        max_step = 0.1 * domain_range.max()

        while calls < self.budget:
            improved = False
            for i in range(self.dim):
                if calls >= self.budget:
                    break
                # positive step
                x_new = best_x.copy()
                x_new[i] = np.clip(best_x[i] + step_size, lb[i], ub[i])
                val = func(x_new)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)
                    break
                # negative step
                x_new = best_x.copy()
                x_new[i] = np.clip(best_x[i] - step_size, lb[i], ub[i])
                val = func(x_new)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)
                    break
            if improved:
                step_size *= 1.2
            else:
                step_size *= 0.5
            step_size = np.clip(step_size, min_step, max_step)

        return best_val, best_x