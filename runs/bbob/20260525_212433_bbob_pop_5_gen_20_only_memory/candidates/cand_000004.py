import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_val = None
        self.n_eval = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point at center
        x = (lb + ub) / 2.0
        val = func(x)
        self.n_eval = 1
        self.best_x = x.copy()
        self.best_val = val
        report_best(self.best_val, self.best_x)

        step = 0.5 * np.mean(ub - lb)  # initial scalar step size

        while self.n_eval < self.budget:
            order = np.arange(self.dim)
            self.rng.shuffle(order)
            improved = False
            for d in order:
                if self.n_eval >= self.budget:
                    break
                # try positive step
                step_size = step * (ub[d] - lb[d])
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step_size, lb[d], ub[d])
                if np.abs(x_new[d] - x[d]) < 1e-15:
                    continue
                val_new = func(x_new)
                self.n_eval += 1
                if val_new < self.best_val:
                    self.best_val = val_new
                    self.best_x = x_new.copy()
                    x = x_new
                    improved = True
                    step *= 2.0
                    report_best(self.best_val, self.best_x)
                    break
                # try negative step
                x_new[d] = np.clip(x[d] - step_size, lb[d], ub[d])
                if np.abs(x_new[d] - x[d]) < 1e-15:
                    continue
                val_new = func(x_new)
                self.n_eval += 1
                if val_new < self.best_val:
                    self.best_val = val_new
                    self.best_x = x_new.copy()
                    x = x_new
                    improved = True
                    step *= 2.0
                    report_best(self.best_val, self.best_x)
                    break
            if not improved:
                step *= 0.5
                # occasional random jump
                if self.rng.rand() < 0.1 and self.n_eval < self.budget:
                    x_new = self.rng.uniform(lb, ub)
                    val_new = func(x_new)
                    self.n_eval += 1
                    if val_new < self.best_val:
                        self.best_val = val_new
                        self.best_x = x_new.copy()
                        x = x_new
                        report_best(self.best_val, self.best_x)
                        step = 0.5 * np.mean(ub - lb)  # reset step
        return self.best_val, self.best_x