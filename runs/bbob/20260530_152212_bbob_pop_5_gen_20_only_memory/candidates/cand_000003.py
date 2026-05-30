import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.stagnation_limit = max(1, int(budget / 20))
        self.counter = 0
        self.last_improvement = 0
        self.best_x = None
        self.best_value = None
        self.scale = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        # Initial scale: 10% of the range per dimension
        self.scale = 0.1 * (ub - lb)
        # Initial random point
        self.best_x = lb + rng.rand(dim) * (ub - lb)
        self.best_value = func(self.best_x)
        self.counter = 1
        self.last_improvement = self.counter
        report_best(self.best_value, self.best_x)
        
        while self.counter < self.budget:
            # Determine if stagnation leads to restart
            if self.counter - self.last_improvement >= self.stagnation_limit:
                # Restart: new random point
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_val = func(new_x)
                self.counter += 1
                if new_val < self.best_value:
                    self.best_value = new_val
                    self.best_x = new_x
                    report_best(self.best_value, self.best_x)
                    self.last_improvement = self.counter
                # Reset scale? Keep as is.
            else:
                # Sample around best
                perturbation = rng.randn(dim) * self.scale
                new_x = self.best_x + perturbation
                new_x = np.clip(new_x, lb, ub)
                new_val = func(new_x)
                self.counter += 1
                if new_val < self.best_value:
                    self.best_value = new_val
                    self.best_x = new_x
                    report_best(self.best_value, self.best_x)
                    self.last_improvement = self.counter
                    # Increase scale on improvement (exploration)
                    self.scale *= 1.2
                else:
                    # Decrease scale on failure (exploitation)
                    self.scale *= 0.95
            # Ensure scale doesn't become too small
            self.scale = np.maximum(self.scale, 1e-10 * (ub - lb))
        return self.best_value, self.best_x