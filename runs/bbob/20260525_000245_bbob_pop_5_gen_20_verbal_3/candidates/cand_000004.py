import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.initial_step = 0.1  # fraction of bound range

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        # Initial random point within bounds
        x = lb + self.rng.random(dim) * (ub - lb)
        best_x = x.copy()
        best_val = func(x)
        report_best(best_val, best_x)
        calls = 1

        step = self.initial_step * (ub - lb).mean()  # adaptive step size
        min_step = 1e-12  # stopping criterion

        while calls < self.budget and step > min_step:
            improved = False
            # Coordinate search along each dimension
            for i in range(dim):
                if calls >= self.budget:
                    break
                # Positive step
                x_new = best_x.copy()
                x_new[i] = np.clip(best_x[i] + step, lb[i], ub[i])
                val_new = func(x_new)
                calls += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)
                    continue  # skip negative if we already improved
                # Negative step
                x_new[i] = np.clip(best_x[i] - step, lb[i], ub[i])
                val_new = func(x_new)
                calls += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)

            # Random pattern step if no improvement from coordinate search
            if not improved and calls < self.budget:
                # Random direction
                direction = self.rng.normal(size=dim)
                direction = direction / np.linalg.norm(direction) * step
                x_new = np.clip(best_x + direction, lb, ub)
                val_new = func(x_new)
                calls += 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    improved = True
                    report_best(best_val, best_x)

            if not improved:
                step *= 0.5  # reduce step if no improvement

        return best_val, best_x