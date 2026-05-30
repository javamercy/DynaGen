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
        # initial point random in bounds
        x0 = np.random.uniform(lb, ub)
        val0 = func(x0)
        self.num_calls = 1
        self.best_x = x0.copy()
        self.best_value = val0
        report_best(val0, x0)
        
        range_max = np.max(ub - lb)
        radius = 0.5 * range_max
        min_radius = 1e-6 * range_max
        decay = 0.9
        no_improve_steps = 0
        max_no_improve = max(1, int(0.1 * self.budget))
        
        while self.num_calls < self.budget:
            if no_improve_steps >= max_no_improve:
                # restart
                candidate = np.random.uniform(lb, ub)
                radius = 0.5 * range_max
                no_improve_steps = 0
            else:
                # perturbation
                candidate = self.best_x + radius * np.random.randn(self.dim)
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
                radius = max(min_radius, radius * decay)
        
        return self.best_value, self.best_x