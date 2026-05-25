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
        best_val = np.inf
        best_x = None
        # Main loop: restart until budget exhausted
        while self.budget > 0:
            # Initialize a new run
            x = lb + self.rng.random(dim) * (ub - lb)
            val = func(x)
            self.budget -= 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            step = 0.2 * (ub - lb)
            min_step = 1e-3 * (ub - lb)
            # Coordinate search within this run
            while self.budget > 0:
                improved = False
                perm = self.rng.permutation(dim)
                for i in perm:
                    if self.budget <= 0:
                        break
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    self.budget -= 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        improved = True
                        continue  # skip negative if positive improved
                    # Try negative direction
                    if self.budget <= 0:
                        break
                    x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    self.budget -= 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        improved = True
                if not improved:
                    step *= 0.5
                    # Check for restart condition: step too small
                    if np.all(step < min_step):
                        break  # exit inner while to restart
        return best_val, best_x