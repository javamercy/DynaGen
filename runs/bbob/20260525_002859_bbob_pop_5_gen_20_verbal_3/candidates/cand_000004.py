import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_f = None
        self.fevals = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial random point
        x0 = lb + self.rng.rand(self.dim) * (ub - lb)
        f0 = func(x0)
        self.fevals = 1
        self.best_x = x0.copy()
        self.best_f = f0
        report_best(self.best_f, self.best_x)
        # initial step sizes
        step = 0.1 * (ub - lb)
        # main loop
        while self.fevals < self.budget:
            # coordinate search cycle
            for i in range(self.dim):
                if self.fevals >= self.budget:
                    break
                # try positive direction
                x_new = self.best_x.copy()
                x_new[i] += step[i]
                x_new = np.clip(x_new, lb, ub)
                f_new = func(x_new)
                self.fevals += 1
                if f_new < self.best_f:
                    self.best_x = x_new.copy()
                    self.best_f = f_new
                    report_best(self.best_f, self.best_x)
                    step[i] *= 1.2  # increase step
                else:
                    # try negative direction
                    x_new = self.best_x.copy()
                    x_new[i] -= step[i]
                    x_new = np.clip(x_new, lb, ub)
                    f_new = func(x_new)
                    self.fevals += 1
                    if f_new < self.best_f:
                        self.best_x = x_new.copy()
                        self.best_f = f_new
                        report_best(self.best_f, self.best_x)
                        step[i] *= 1.2
                    else:
                        # no improvement, shrink step
                        step[i] *= 0.5
                # prevent step from becoming too small
                if step[i] < 1e-15:
                    step[i] = 1e-15
            # random direction search
            if self.fevals < self.budget:
                dir = self.rng.randn(self.dim)
                norm = np.linalg.norm(dir)
                if norm > 0:
                    dir = dir / norm
                step_mean = np.mean(step)
                x_new = self.best_x + step_mean * dir
                x_new = np.clip(x_new, lb, ub)
                f_new = func(x_new)
                self.fevals += 1
                if f_new < self.best_f:
                    self.best_x = x_new.copy()
                    self.best_f = f_new
                    report_best(self.best_f, self.best_x)
        return self.best_f, self.best_x