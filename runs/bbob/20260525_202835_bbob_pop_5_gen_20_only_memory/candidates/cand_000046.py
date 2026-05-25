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
        step = 0.1 * (ub - lb)  # per-coordinate step sizes
        improvement = True
        while self.budget > 0:
            if not improvement:
                step *= 0.5
                improvement = True
            improved = False
            cycle_start_x = best_x.copy()
            perm = self.rng.permutation(dim)
            for i in perm:
                if self.budget <= 0:
                    break
                # positive direction
                x_new = best_x.copy()
                x_new[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                val_new = func(x_new)
                self.budget -= 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                    step[i] *= 1.5
                    improved = True
                    continue
                # negative direction
                if self.budget <= 0:
                    break
                x_new = best_x.copy()
                x_new[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                val_new = func(x_new)
                self.budget -= 1
                if val_new < best_val:
                    best_val = val_new
                    best_x = x_new.copy()
                    report_best(best_val, best_x)
                    step[i] *= 1.5
                    improved = True
            # line search along displacement if cycle improved
            if improved and self.budget > 0:
                delta = best_x - cycle_start_x
                if np.linalg.norm(delta) > 0:
                    x_line = best_x + 1.5 * delta
                    x_line = np.clip(x_line, lb, ub)
                    val_line = func(x_line)
                    self.budget -= 1
                    if val_line < best_val:
                        best_val = val_line
                        best_x = x_line.copy()
                        report_best(best_val, best_x)
                        step *= 1.2  # increase step sizes globally
            if not improved:
                improvement = False
        return best_val, best_x