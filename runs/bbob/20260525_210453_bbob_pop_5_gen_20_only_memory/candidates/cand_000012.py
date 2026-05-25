import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_val = None
        self.evals_used = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial point
        x = lb + self.rng.rand(self.dim) * (ub - lb)
        val = func(x)
        self.evals_used += 1
        self._update_best(val, x)
        step = 0.1 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        no_improve_iter = 0
        max_no_improve = 10 * self.dim
        while self.evals_used < self.budget:
            x_prev = x.copy()
            improved = False
            # Coordinate cycle
            for i in range(self.dim):
                if self.evals_used >= self.budget:
                    break
                x_new = x.copy()
                x_new[i] += step[i]
                x_new[i] = np.clip(x_new[i], lb[i], ub[i])
                if self.evals_used < self.budget:
                    val_new = func(x_new)
                    self.evals_used += 1
                    if val_new < self.best_val:
                        self._update_best(val_new, x_new)
                        x = x_new
                        step[i] *= 1.2
                        improved = True
                        continue
                x_new = x.copy()
                x_new[i] -= step[i]
                x_new[i] = np.clip(x_new[i], lb[i], ub[i])
                if self.evals_used < self.budget:
                    val_new = func(x_new)
                    self.evals_used += 1
                    if val_new < self.best_val:
                        self._update_best(val_new, x_new)
                        x = x_new
                        step[i] *= 1.2
                        improved = True
                        continue
                step[i] *= 0.9
                if step[i] < min_step[i]:
                    step[i] = min_step[i]
            # Pattern move
            if improved and self.evals_used < self.budget:
                direction = x - x_prev
                if np.linalg.norm(direction) > 0:
                    factor = 1.0
                    x_pattern = x + factor * direction
                    x_pattern = np.clip(x_pattern, lb, ub)
                    if self.evals_used < self.budget:
                        val_pattern = func(x_pattern)
                        self.evals_used += 1
                        if val_pattern < self.best_val:
                            self._update_best(val_pattern, x_pattern)
                            x = x_pattern
                            improved = True
            # Restart if no improvement
            if not improved:
                no_improve_iter += 1
                if no_improve_iter >= max_no_improve and self.evals_used < self.budget:
                    # Restart
                    x = lb + self.rng.rand(self.dim) * (ub - lb)
                    step = 0.1 * (ub - lb)
                    no_improve_iter = 0
                    # Evaluate new point only if not exceeding budget
                    if self.evals_used < self.budget:
                        val = func(x)
                        self.evals_used += 1
                        self._update_best(val, x)
            else:
                no_improve_iter = 0
        return self.best_val, self.best_x

    def _update_best(self, val, x):
        if self.best_val is None or val < self.best_val:
            self.best_val = val
            self.best_x = x.copy()
            report_best(val, x)