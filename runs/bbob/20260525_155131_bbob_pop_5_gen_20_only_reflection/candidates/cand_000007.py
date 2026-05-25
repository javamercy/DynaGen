import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_val = np.inf
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial point
        self.best_x = self.rng.uniform(lb, ub, size=self.dim)
        self.best_val = func(self.best_x)
        self.calls += 1
        report_best(self.best_val, self.best_x)

        # Parameters
        init_step = 0.2 * (ub - lb)
        step = init_step.copy()
        min_step = 1e-12 * np.ones(self.dim)
        stagnation_limit = max(10, int(0.02 * self.budget))
        no_improve = 0

        while self.calls < self.budget:
            # Check for restart
            if no_improve >= stagnation_limit:
                self.best_x = self.rng.uniform(lb, ub, size=self.dim)
                self.best_val = func(self.best_x)
                self.calls += 1
                report_best(self.best_val, self.best_x)
                step = init_step.copy()
                no_improve = 0
                if self.calls >= self.budget:
                    break

            # Pattern search: random coordinate and direction
            coord = self.rng.integers(0, self.dim)
            direction = 1 if self.rng.uniform() < 0.5 else -1
            x_candidate = self.best_x.copy()
            x_candidate[coord] += direction * step[coord]
            x_candidate = np.clip(x_candidate, lb, ub)
            val = func(x_candidate)
            self.calls += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = x_candidate.copy()
                report_best(self.best_val, self.best_x)
                no_improve = 0
                step[coord] = min(step[coord] * 1.5, ub[coord] - lb[coord])
            else:
                no_improve += 1
                step[coord] = max(step[coord] * 0.8, min_step[coord])

        return self.best_val, self.best_x