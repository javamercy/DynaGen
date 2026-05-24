import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        ranges = ub - lb

        # initial random point
        x = lb + self.rng.rand(self.dim) * ranges
        y = func(x)
        best_x = x.copy()
        best_y = y
        report_best(best_y, best_x)

        calls = 1
        center = best_x.copy()
        sigma = 0.2  # initial fraction of range
        max_patience = max(10, self.dim * 2)
        patience = 0

        while calls < self.budget:
            if sigma > 0.5:
                sigma = 0.5
            if sigma < 1e-10:
                sigma = 1e-10

            # sample candidate
            candidate = center + sigma * ranges * self.rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            new_y = func(candidate)
            calls += 1

            if new_y < best_y:
                best_y = new_y
                best_x = candidate.copy()
                report_best(best_y, best_x)
                center = candidate.copy()
                sigma *= 1.2  # expand
                patience = 0
            else:
                patience += 1
                sigma *= 0.95  # shrink
                if patience >= max_patience:
                    # restart
                    x = lb + self.rng.rand(self.dim) * ranges
                    if calls < self.budget:
                        y = func(x)
                        calls += 1
                        if y < best_y:
                            best_y = y
                            best_x = x.copy()
                            report_best(best_y, best_x)
                        center = x.copy()
                        sigma = 0.2
                        patience = 0

        return best_y, best_x