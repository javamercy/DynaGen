import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.evals = 0

    def _evaluate(self, func, x, best_val, best_x):
        val = func(x)
        self.evals += 1
        if val < best_val:
            best_val = val
            best_x = x.copy()
            report_best(best_val, best_x)
        return val, best_val, best_x

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        x = lb + self.rng.rand(self.dim) * (ub - lb)
        val = func(x)
        self.evals = 1
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)
        
        # step size: 5% of range per dimension
        step = 0.05 * (ub - lb)
        min_step = 1e-10 * (ub - lb)
        x_base = x.copy()
        val_base = val
        
        while self.evals < self.budget:
            improved = False
            x_old = x_base.copy()
            # exploratory moves
            for i in range(self.dim):
                if self.evals >= self.budget:
                    break
                # positive direction
                x_new = x_base.copy()
                x_new[i] = np.clip(x_base[i] + step[i], lb[i], ub[i])
                val_new, best_val, best_x = self._evaluate(func, x_new, best_val, best_x)
                if val_new < val_base:
                    val_base = val_new
                    x_base = x_new.copy()
                    improved = True
                    continue
                # negative direction
                x_new[i] = np.clip(x_base[i] - step[i], lb[i], ub[i])
                val_new, best_val, best_x = self._evaluate(func, x_new, best_val, best_x)
                if val_new < val_base:
                    val_base = val_new
                    x_base = x_new.copy()
                    improved = True
            if not improved:
                # reduce step size
                step = step * 0.5
                if np.all(step < min_step):
                    break
            else:
                # pattern step
                if self.evals < self.budget:
                    x_pattern = np.clip(2 * x_base - x_old, lb, ub)
                    val_pattern, best_val, best_x = self._evaluate(func, x_pattern, best_val, best_x)
                    if val_pattern < val_base:
                        val_base = val_pattern
                        x_base = x_pattern.copy()
        return best_val, best_x