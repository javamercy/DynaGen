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
        dim = self.dim
        # Initial random point
        best_x = lb + self.rng.random(dim) * (ub - lb)
        best_val = func(best_x)
        self.budget -= 1
        report_best(best_val, best_x)
        step = 0.1 * (ub - lb)  # per-coordinate step sizes
        while self.budget > 0:
            # Determine number of samples for this generation
            lam = min(10 * dim, self.budget)
            candidates = np.zeros((lam, dim))
            for i in range(lam):
                candidates[i] = np.clip(
                    best_x + step * self.rng.normal(0, 1, dim),
                    lb,
                    ub
                )
            # Evaluate candidates
            improved = False
            for i in range(lam):
                if self.budget <= 0:
                    break
                val = func(candidates[i])
                self.budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = candidates[i].copy()
                    report_best(best_val, best_x)
                    improved = True
            # Adapt step sizes
            if improved:
                step *= 1.2
            else:
                step *= 0.5
            # Ensure step sizes stay within reasonable bounds
            step = np.clip(step, 1e-8 * (ub - lb), 0.5 * (ub - lb))
        return best_val, best_x