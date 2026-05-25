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
        # initial random point
        best_x = lb + self.rng.random(dim) * (ub - lb)
        best_val = func(best_x)
        self.budget -= 1
        report_best(best_val, best_x)
        step = 0.1 * (ub - lb)  # per-coordinate step sizes
        while self.budget > 0:
            improved = False
            # coordinate pattern poll
            for i in range(dim):
                if self.budget <= 0:
                    break
                for direction in [1, -1]:
                    if self.budget <= 0:
                        break
                    x_new = best_x.copy()
                    x_new[i] = np.clip(best_x[i] + direction * step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    self.budget -= 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new
                        report_best(best_val, best_x)
                        step[i] *= 1.2
                        improved = True
                        break  # break negative direction after improvement
                if improved:
                    break  # restart poll with first coordinate
            if improved:
                continue
            # random perturbations if no coordinate improvement
            if self.budget > 0:
                for _ in range(min(dim, self.budget)):
                    x_new = np.clip(best_x + step * self.rng.uniform(-1, 1, dim), lb, ub)
                    val_new = func(x_new)
                    self.budget -= 1
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new
                        report_best(best_val, best_x)
                        improved = True
                        break
                if improved:
                    step *= 1.1
                    continue
            # reduce step if no improvement
            step *= 0.5
        return best_val, best_x