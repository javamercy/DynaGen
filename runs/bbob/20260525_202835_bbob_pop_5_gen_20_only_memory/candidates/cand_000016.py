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
        # Multi-point initialization: sample a few points and pick best
        num_initial = max(2, min(10, self.budget // 20))
        num_initial = min(num_initial, self.budget)
        best_val = np.inf
        best_x = None
        for _ in range(num_initial):
            x = lb + self.rng.random(dim) * (ub - lb)
            val = func(x)
            self.budget -= 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
            report_best(best_val, best_x)
        # Proceed with coordinate search from best initial point
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
                    continue
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