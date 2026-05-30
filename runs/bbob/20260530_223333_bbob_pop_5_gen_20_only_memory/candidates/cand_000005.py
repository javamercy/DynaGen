import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_value = np.inf
        best_x = None
        calls = 0

        # Initial global sampling
        n_init = min(50, max(2, int(self.budget * 0.2)))
        for _ in range(n_init):
            if calls >= self.budget:
                break
            x = self.rng.uniform(lb, ub, size=self.dim)
            val = func(x)
            calls += 1
            if val < best_value:
                best_value = val
                best_x = x.copy()
                report_best(best_value, best_x)

        # Local refinement
        sigma = np.mean(ub - lb) * 0.2
        while calls < self.budget:
            # Random restart periodically
            if calls % max(1, int(self.budget / 5)) == 0:
                x = self.rng.uniform(lb, ub, size=self.dim)
            else:
                # Perturb best
                x = best_x + self.rng.normal(0, sigma, size=self.dim)
            x = np.clip(x, lb, ub)
            val = func(x)
            calls += 1
            if val < best_value:
                best_value = val
                best_x = x.copy()
                report_best(best_value, best_x)
            sigma *= 0.99

        return (best_value, best_x)