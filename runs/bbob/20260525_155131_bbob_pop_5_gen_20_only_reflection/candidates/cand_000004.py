import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_val = np.inf

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial random point
        x0 = lb + self.rng.rand(self.dim) * (ub - lb)
        val0 = func(x0)
        self.best_x = x0.copy()
        self.best_val = val0
        report_best(self.best_val, self.best_x)
        evals = 1
        # initial step size as fraction of range
        step_size = 0.1
        while evals < self.budget:
            improved = False
            for i in range(self.dim):
                if evals >= self.budget:
                    break
                step = step_size * (ub[i] - lb[i])
                # positive step
                x_new = self.best_x.copy()
                x_new[i] = np.clip(self.best_x[i] + step, lb[i], ub[i])
                val_new = func(x_new)
                evals += 1
                if val_new < self.best_val:
                    self.best_val = val_new
                    self.best_x[i] = x_new[i]
                    improved = True
                    report_best(self.best_val, self.best_x)
                else:
                    if evals >= self.budget:
                        break
                    # negative step
                    x_new2 = self.best_x.copy()
                    x_new2[i] = np.clip(self.best_x[i] - step, lb[i], ub[i])
                    val_new2 = func(x_new2)
                    evals += 1
                    if val_new2 < self.best_val:
                        self.best_val = val_new2
                        self.best_x[i] = x_new2[i]
                        improved = True
                        report_best(self.best_val, self.best_x)
            # adapt step size
            if improved:
                step_size *= 1.2
            else:
                step_size *= 0.5
            # random perturbation
            if evals < self.budget:
                x_rnd = np.clip(self.best_x + step_size * (ub - lb) * (2*self.rng.rand(self.dim)-1), lb, ub)
                val_rnd = func(x_rnd)
                evals += 1
                if val_rnd < self.best_val:
                    self.best_val = val_rnd
                    self.best_x = x_rnd.copy()
                    improved = True
                    report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x