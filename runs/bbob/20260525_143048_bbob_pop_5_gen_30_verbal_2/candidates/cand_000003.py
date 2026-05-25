import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # initial point
        x = rng.uniform(lb, ub, size=dim)
        best_x = x.copy()
        best_val = func(x)
        calls = 1
        report_best(best_val, best_x)

        # adaptive parameters
        radius = (ub - lb).mean() / 2
        min_radius = 1e-8
        factor = 0.9
        patience = int(0.1 * budget) + 1
        fail_streak = 0

        while calls < budget:
            # sample candidate
            candidate = rng.normal(loc=best_x, scale=radius, size=dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            calls += 1

            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                fail_streak = 0
                radius /= factor
            else:
                fail_streak += 1
                if fail_streak >= patience or radius < min_radius:
                    # restart
                    x = rng.uniform(lb, ub, size=dim)
                    val = func(x)
                    calls += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                    radius = (ub - lb).mean() / 2
                    fail_streak = 0
                else:
                    radius *= factor

            # ensure not overshoot budget
            if calls >= budget:
                break

        return best_val, best_x