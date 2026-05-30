import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_value = np.inf

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial point
        x0 = lb + self.rng.rand(self.dim) * (ub - lb)
        val0 = func(x0)
        self.best_x = x0.copy()
        self.best_value = val0
        report_best(self.best_value, self.best_x)
        evals = 1
        # Initial step sizes: 10% of range per dimension
        step_sizes = (ub - lb) * 0.1
        while evals < self.budget:
            # Interleave coordinate and pattern search
            if self.rng.rand() < 0.5 or self.dim == 1:
                # Coordinate search
                d = self.rng.randint(self.dim)
                step = step_sizes[d]
                # Positive direction
                x_candidate = self.best_x.copy()
                x_candidate[d] = np.clip(self.best_x[d] + step, lb[d], ub[d])
                val_candidate = func(x_candidate)
                evals += 1
                if val_candidate < self.best_value:
                    self.best_x = x_candidate
                    self.best_value = val_candidate
                    report_best(self.best_value, self.best_x)
                    step_sizes[d] *= 1.2
                else:
                    # Negative direction
                    x_candidate = self.best_x.copy()
                    x_candidate[d] = np.clip(self.best_x[d] - step, lb[d], ub[d])
                    val_candidate = func(x_candidate)
                    evals += 1
                    if val_candidate < self.best_value:
                        self.best_x = x_candidate
                        self.best_value = val_candidate
                        report_best(self.best_value, self.best_x)
                        step_sizes[d] *= 1.2
                    else:
                        step_sizes[d] *= 0.5
                step_sizes[d] = max(step_sizes[d], 1e-10)
            else:
                # Pattern search: random direction
                direction = self.rng.randn(self.dim)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.ones(self.dim) / np.sqrt(self.dim)
                else:
                    direction /= norm
                avg_step = np.mean(step_sizes)
                candidate = np.clip(self.best_x + avg_step * direction, lb, ub)
                val_candidate = func(candidate)
                evals += 1
                if val_candidate < self.best_value:
                    self.best_x = candidate
                    self.best_value = val_candidate
                    report_best(self.best_value, self.best_x)
                    step_sizes *= 1.2
                else:
                    step_sizes *= 0.5
                step_sizes = np.maximum(step_sizes, 1e-10)
            if evals >= self.budget:
                break
        return self.best_value, self.best_x