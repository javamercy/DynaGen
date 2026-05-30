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
        self.exploration_prob = 0.05
        self.growth_factor = 3.0
        self.shrink_factor = 0.3
        self.stagnation_limit = 10
        self.min_radius = 1e-5
        self.max_radius = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_size = ub - lb
        self.max_radius = 0.5 * np.max(range_size)
        self.radius = 0.05 * self.max_radius

        x0 = lb + self.rng.uniform(0, 1, size=self.dim) * range_size
        f0 = func(x0)
        self.calls += 1
        self.best_x = x0.copy()
        self.best_value = f0
        report_best(self.best_value, self.best_x)

        while self.calls < self.budget:
            if self.rng.uniform() < self.exploration_prob:
                new_x = lb + self.rng.uniform(0, 1, size=self.dim) * range_size
                new_val = func(new_x)
                self.calls += 1
                if new_val < self.best_value:
                    self.best_x = new_x.copy()
                    self.best_value = new_val
                    report_best(self.best_value, self.best_x)
            else:
                perturb = self.rng.uniform(-self.radius, self.radius, size=self.dim)
                new_x = self.best_x + perturb
                new_x = np.clip(new_x, lb, ub)
                new_val = func(new_x)
                self.calls += 1
                if new_val < self.best_value:
                    # line search along improvement direction
                    direction = new_x - self.best_x
                    step = 1.0
                    candidate = self.best_x + step * direction
                    candidate = np.clip(candidate, lb, ub)
                    best_candidate = candidate.copy()
                    best_candidate_val = func(candidate)
                    self.calls += 1
                    # try larger steps while improvement
                    while self.calls < self.budget:
                        step *= 2.0
                        candidate = self.best_x + step * direction
                        candidate = np.clip(candidate, lb, ub)
                        val = func(candidate)
                        self.calls += 1
                        if val < best_candidate_val:
                            best_candidate_val = val
                            best_candidate = candidate.copy()
                        else:
                            break
                    # try smaller steps if first step not best? Alternatively, keep best.
                    # Now update best if line search found better
                    if best_candidate_val < self.best_value:
                        self.best_x = best_candidate.copy()
                        self.best_value = best_candidate_val
                        report_best(self.best_value, self.best_x)
                    else:
                        # if line search didn't improve, the original new_x was better, but we already checked that new_val < best_value.
                        # Actually new_val is less than best_value, so we use new_x as best.
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

            if self.radius <= self.min_radius or self.stagnation >= 2 * self.stagnation_limit:
                if self.calls >= self.budget:
                    break
                x_new = lb + self.rng.uniform(0, 1, size=self.dim) * range_size
                f_new = func(x_new)
                self.calls += 1
                if f_new < self.best_value:
                    self.best_x = x_new.copy()
                    self.best_value = f_new
                    report_best(self.best_value, self.best_x)
                self.radius = 0.05 * self.max_radius
                self.stagnation = 0

        return self.best_value, self.best_x