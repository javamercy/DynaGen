import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget == 0:
            return float('inf'), None
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        x = lb + (ub - lb) * self.rng.rand(self.dim)
        val = func(x)
        self.best_val = val
        self.best_x = x.copy()
        report_best(self.best_val, self.best_x)
        evals = 1
        # parameters
        T0 = 1.0
        T = T0
        step_size = 0.2 * (ub - lb)  # per dimension
        n_adapt = max(1, self.budget // 20)  # adapt every n_adapt evaluations
        success_count = 0
        while evals < self.budget:
            # generate neighbor
            delta = self.rng.randn(self.dim) * step_size
            y = x + delta
            y = np.clip(y, lb, ub)
            new_val = func(y)
            evals += 1
            if new_val < val:
                accept = True
            else:
                if self.rng.rand() < np.exp((val - new_val) / T):
                    accept = True
                else:
                    accept = False
            if accept:
                x = y
                val = new_val
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
                success_count += 1
            # adapt step size
            if evals % n_adapt == 0:
                success_rate = success_count / n_adapt
                if success_rate > 0.2:
                    step_size *= 1.2
                else:
                    step_size *= 0.85
                success_count = 0
            # cool down
            T = T0 * (1 - evals / self.budget)
        return self.best_val, self.best_x