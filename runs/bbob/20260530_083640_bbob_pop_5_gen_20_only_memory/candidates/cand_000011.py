import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.best_x = None
        self.best_value = np.inf
        self.num_calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb
        step_size = 0.1 * range_
        max_no_improve = max(1, int(0.1 * self.budget))
        no_improve_steps = 0

        # Initial point
        x0 = np.random.uniform(lb, ub)
        val0 = func(x0)
        self.num_calls = 1
        self.best_x = x0.copy()
        self.best_value = val0
        report_best(val0, x0)

        while self.num_calls < self.budget:
            if no_improve_steps >= max_no_improve:
                # Restart from uniform random
                candidate = np.random.uniform(lb, ub)
                no_improve_steps = 0
            else:
                # Perturb from best
                candidate = self.best_x + step_size * np.random.randn(self.dim)
                candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            self.num_calls += 1
            if val < self.best_value:
                self.best_value = val
                self.best_x = candidate.copy()
                report_best(val, candidate)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        return self.best_value, self.best_x