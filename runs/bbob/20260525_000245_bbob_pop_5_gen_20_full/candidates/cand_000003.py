import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_value = np.inf
        self.best_x = None
        self.radius = 0.1  # initial radius, will be scaled by bounds range

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget_remaining = self.budget

        # Initial uniform point
        x = self.rng.uniform(lb, ub, size=dim)
        f = func(x)
        budget_remaining -= 1
        if f < self.best_value:
            self.best_value = f
            self.best_x = x.copy()
            self._report_best(f, x)

        patience = max(1, 2 * dim)
        no_improve_count = 0

        while budget_remaining > 0:
            # Generate candidate around best with adaptive radius
            perturbation = self.rng.normal(0, self.radius, size=dim)
            candidate = self.best_x + perturbation * (ub - lb)
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            budget_remaining -= 1

            if f < self.best_value:
                self.best_value = f
                self.best_x = candidate.copy()
                self._report_best(f, candidate)
                self.radius *= 1.2  # expand radius on success
                no_improve_count = 0
            else:
                self.radius *= 0.8  # shrink radius on failure
                no_improve_count += 1

            # Restart if stagnation
            if no_improve_count >= patience:
                x = self.rng.uniform(lb, ub, size=dim)
                f = func(x)
                budget_remaining -= 1
                if f < self.best_value:
                    self.best_value = f
                    self.best_x = x.copy()
                    self._report_best(f, x)
                self.radius = 0.1
                no_improve_count = 0

        return (self.best_value, self.best_x)

    def _report_best(self, value, x):
        try:
            report_best(value, x)
        except NameError:
            pass  # report_best not available outside benchmark