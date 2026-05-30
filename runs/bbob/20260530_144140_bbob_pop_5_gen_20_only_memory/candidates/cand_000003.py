import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_value = np.inf
        self.radius = None
        self.calls = 0
        self.stagnation = 0
        self.restart_count = 0
        # adaptation parameters
        self.growth_factor = 2.0
        self.shrink_factor = 0.5
        self.stagnation_limit = 10
        self.min_radius = 1e-5
        self.max_radius = None

    def __call__(self, func):
        # get bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        # compute initial radius as 10% of the range
        range_size = ub - lb
        self.max_radius = 0.5 * np.max(range_size)
        self.radius = 0.1 * self.max_radius

        # first point: random within bounds
        x0 = lb + self.rng.uniform(0, 1, size=self.dim) * range_size
        f0 = func(x0)
        self.calls += 1
        self.best_x = x0.copy()
        self.best_value = f0
        report_best(self.best_value, self.best_x)

        while self.calls < self.budget:
            # sample new point in hypercube around best_x
            perturb = self.rng.uniform(-self.radius, self.radius, size=self.dim)
            new_x = self.best_x + perturb
            # clip to bounds
            new_x = np.clip(new_x, lb, ub)
            # evaluate
            new_val = func(new_x)
            self.calls += 1

            if new_val < self.best_value:
                self.best_x = new_x.copy()
                self.best_value = new_val
                report_best(self.best_value, self.best_x)
                self.radius = min(self.max_radius, self.radius * self.growth_factor)
                self.stagnation = 0
            else:
                self.stagnation += 1
                if self.stagnation >= self.stagnation_limit:
                    self.radius = max(self.min_radius, self.radius * self.shrink_factor)
                    self.stagnation = 0
            # restart if radius too small or too many iterations without improvement
            if self.radius <= self.min_radius or self.stagnation >= 2 * self.stagnation_limit:
                # random restart
                x_new = lb + self.rng.uniform(0, 1, size=self.dim) * range_size
                f_new = func(x_new)
                self.calls += 1
                if f_new < self.best_value:
                    self.best_x = x_new.copy()
                    self.best_value = f_new
                    report_best(self.best_value, self.best_x)
                self.radius = 0.1 * self.max_radius
                self.stagnation = 0

        return self.best_value, self.best_x